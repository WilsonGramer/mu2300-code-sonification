use itertools::Itertools;
use midly::{num::*, Header, MidiMessage, Smf, TrackEvent, TrackEventKind};
use rust_music_theory::{
    chord::{Chord, Number, Quality},
    note::{Notes, PitchClass},
};
use std::{
    collections::{hash_map::DefaultHasher, HashMap, HashSet, VecDeque},
    hash::{Hash, Hasher},
    io::{self, Read},
    ops::Range,
    rc::Rc,
};
use swc_common::sync::Lrc;
use swc_common::{FileName, SourceMap};
use swc_ecma_ast::{EsVersion, Expr, Stmt};
use swc_ecma_parser::{lexer::Lexer, Parser, StringInput, Syntax};
use swc_ecma_visit::{Visit, VisitWith};

const TRACK_0_CAP: usize = 4;
const TRACK_1_CAP: usize = 3;
const TRACK_2_CAP: usize = 2;
const COMPLEXITY_CAP: u64 = 32;
const CHORD_CAP: u64 = 3;
const VELOCITY: u7 = u7::from_int_lossy(100);

#[derive(Debug, Clone)]
struct Sequence(Rc<SequenceInner>);

impl PartialEq for Sequence {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for Sequence {}

impl Hash for Sequence {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (Rc::as_ptr(&self.0) as usize).hash(state);
    }
}

#[derive(Debug)]
struct SequenceInner {
    track: usize,
    notes: Vec<Vec<u8>>,
}

#[derive(Debug)]
enum Instruction {
    Start(Sequence),
    Stop(Sequence),
}

fn main() {
    // Parse the program

    let mut src = String::new();
    std::io::stdin()
        .read_to_string(&mut src)
        .expect("failed to read from stdin");

    let cm = Lrc::<SourceMap>::default();
    let fm = cm.new_source_file(FileName::Anon, src);

    let lexer = Lexer::new(
        Syntax::Es(Default::default()),
        EsVersion::latest(),
        StringInput::from(fm.as_ref()),
        None,
    );

    let mut parser = Parser::new_from(lexer);

    let module = parser.parse_module().expect("failed to parse file");

    // Determine chord information

    let seed = hash(&module);

    let quality = match seed % 4 {
        0 => Quality::Major,
        1 => Quality::Minor,
        2 => Quality::Diminished,
        3 => Quality::Augmented,
        _ => unreachable!(),
    };

    let mut hash_visitor = HashVisitor::default();
    module.visit_children_with(&mut hash_visitor);

    // Generate and remove sequences over time

    let mut active_sequences = (0..3)
        .map(|track| (track, HashSet::<Sequence>::new()))
        .collect::<HashMap<_, _>>();

    let mut instructions = Vec::new();

    let mut statements = VecDeque::from(hash_visitor.statements);
    while let Some(statement) = statements.pop_front() {
        let seed = statement.seed.unwrap_or(seed);
        let hash = statement.hash;
        let complexity = statement.complexity;

        let root = seed % 12;
        let chord = Chord::new(PitchClass::from_u8(root as u8), quality, Number::Triad);

        instructions.push(Vec::new());

        for (&track, sequences) in &mut active_sequences {
            let cap = match track {
                0 => TRACK_0_CAP,
                1 => TRACK_1_CAP,
                2 => TRACK_2_CAP,
                _ => unreachable!(),
            };

            if sequences.len() > cap {
                if let Some(sequence) = sequences.iter().next().cloned() {
                    sequences.remove(&sequence);

                    instructions
                        .last_mut()
                        .unwrap()
                        .push(Instruction::Stop(sequence));
                }
            } else {
                let sequence = generate_sequence(hash, complexity, chord.clone());

                sequences.insert(sequence.clone());

                instructions
                    .last_mut()
                    .unwrap()
                    .push(Instruction::Start(sequence));
            }
        }
    }

    // Stop any remaining sequences
    instructions.push(
        active_sequences
            .into_values()
            .flatten()
            .map(Instruction::Stop)
            .collect(),
    );

    // Generate MIDI sequences based on the instructions

    let mut midi = Vec::<(usize, usize, TrackEventKind)>::new();
    let mut time = 0;
    let mut active_sequences = HashSet::new();
    for instruction_group in instructions {
        for instruction in instruction_group {
            match instruction {
                Instruction::Start(sequence) => {
                    active_sequences.insert(sequence);
                }
                Instruction::Stop(sequence) => {
                    active_sequences.remove(&sequence);
                }
            }
        }

        for sequence in &active_sequences {
            let mut time = time;

            for chord in &sequence.0.notes {
                let delta = COMPLEXITY_CAP as usize / sequence.0.notes.len();

                for note in chord {
                    midi.push((
                        sequence.0.track,
                        time,
                        TrackEventKind::Midi {
                            channel: u4::from(0),
                            message: MidiMessage::NoteOn {
                                key: u7::from(*note),
                                vel: u7::from(VELOCITY),
                            },
                        },
                    ));
                }

                time += delta;

                for note in chord {
                    midi.push((
                        sequence.0.track,
                        time,
                        TrackEventKind::Midi {
                            channel: u4::from(0),
                            message: MidiMessage::NoteOn {
                                key: u7::from(*note),
                                vel: u7::from(0),
                            },
                        },
                    ));
                }
            }
        }

        time += COMPLEXITY_CAP as usize;
    }

    midi.sort_unstable_by_key(|(_, time, _)| *time);

    // Export the MIDI as a file

    let mut smf = Smf::new(Header {
        format: midly::Format::Parallel,
        timing: midly::Timing::Metrical(u15::from(1)),
    });

    smf.tracks.extend(vec![Vec::new(); 3]);

    let mut time = 0;
    for (track, event_time, event) in midi {
        let delta = event_time - time;

        let track_event = TrackEvent {
            delta: u28::from(delta as u32),
            kind: event,
        };

        smf.tracks[track].push(track_event);

        time = event_time;
    }

    smf.write_std(io::stdout())
        .expect("failed to write MIDI file");
}

#[derive(Debug, Clone, Copy)]
struct VisitedStmt {
    seed: Option<u64>,
    hash: u64,
    complexity: u64,
}

#[derive(Debug, Default)]
struct HashVisitor {
    statements: Vec<VisitedStmt>,
    current_parent_hash: Option<u64>,
}

impl Visit for HashVisitor {
    fn visit_stmt(&mut self, stmt: &Stmt) {
        let mut complexity_visitor = ComplexityVisitor {
            complexity: Some(0),
        };

        stmt.visit_children_with(&mut complexity_visitor);

        let prev_parent_hash = self.current_parent_hash.replace(hash(stmt));

        if let Some(complexity) = complexity_visitor.complexity {
            self.statements.push(VisitedStmt {
                seed: prev_parent_hash,
                hash: hash((self.current_parent_hash, stmt)),
                complexity: complexity.next_power_of_two().min(COMPLEXITY_CAP),
            });
        }

        stmt.visit_children_with(self);

        self.current_parent_hash = prev_parent_hash;
    }
}

#[derive(Debug)]
struct ComplexityVisitor {
    complexity: Option<u64>,
}

impl Visit for ComplexityVisitor {
    fn visit_expr(&mut self, expr: &Expr) {
        if let Some(complexity) = &mut self.complexity {
            *complexity += 1;
        }

        expr.visit_children_with(self);
    }

    fn visit_stmt(&mut self, _stmt: &Stmt) {
        // Ignore nested statements
        self.complexity = None;
    }
}

fn generate_sequence(hash: u64, complexity: u64, mut chord: Chord) -> Sequence {
    let track = match complexity {
        0 | 1 | 2 => 0,
        4 => 1,
        8 | 16 | COMPLEXITY_CAP => 2,
        _ => unreachable!(),
    };

    chord.octave = choose(1, 4..7, hash).pop().unwrap_or(5) as u8;

    let notes = (0..complexity)
        .map(|n| {
            let number_of_notes = self::hash((hash, n)) % (CHORD_CAP) + 1;

            chord
                .notes()
                .into_iter()
                .map(|note| note.octave * 12 + note.pitch_class.into_u8())
                .take(number_of_notes as usize)
                .collect()
        })
        .collect();

    Sequence(Rc::new(SequenceInner { track, notes }))
}

fn hash(x: impl Hash) -> u64 {
    let mut hasher = DefaultHasher::new();
    x.hash(&mut hasher);
    hasher.finish()
}

/// Choose `n` items from a `range`, randomized using `seed`
fn choose(n: usize, range: Range<u64>, seed: u64) -> Vec<u64> {
    range
        .enumerate()
        .map(|(index, item)| (hash((seed, index)), item))
        .sorted_by_key(|(hash, _)| *hash)
        .map(|(_, item)| item)
        .take(n)
        .collect()
}
