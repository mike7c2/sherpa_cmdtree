//! cmdtree-lint — predict how reliably an ASR will pick up a cmdtree command tree.
//!
//! A grammar-authoring tool, kept out of the runtime recognizer: it scores a
//! [`cmdtree`] grammar for acoustic confusability using espeak phonetics.
//!
//! * [`phonetics`] — espeak G2P + distinctive-feature-weighted phone distance.
//! * [`assess`] — score a grammar's commands, enums, and slot anchors.

pub mod assess;
pub mod phonetics;
