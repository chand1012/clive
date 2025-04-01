use anyhow::{Context, Result};
use rusqlite::{ffi::sqlite3_auto_extension, params, Connection};
use sqlite_vec::sqlite3_vec_init;
use zerocopy::IntoBytes;

use crate::utils::Timestamp;

// Define a trait for the embedding functionality we need
pub trait Embedder {
    fn embed(&mut self, text: &str) -> Result<Vec<f32>>;
    fn batch_embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}

// Implement the trait for the real Llama struct
impl Embedder for crate::utils::Llama {
    fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        self.embed(text)
    }

    fn batch_embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.batch_embed(texts)
    }
}

pub struct DBTimestamp {
    pub id: i64,
    pub start_time: f64,
    pub end_time: f64,
    pub transcript: String,
}

pub struct VectorDB {
    conn: Connection,
}

// maybe in the future if this module gets GPU support we can remove
// the need for a copy of llama that handles the embedding: https://github.com/asg017/sqlite-lembed
// however for now we need our own implementation

// use scalar function method as defined here so we can use a normal table
// https://alexgarcia.xyz/sqlite-vec/features/knn.html#manually-with-sql-scalar-functions

impl VectorDB {
    pub fn new_in_memory(dimensions: usize) -> Result<Self> {
        // initialize an in memory database with the vector extension enabled
        unsafe {
            sqlite3_auto_extension(Some(std::mem::transmute(sqlite3_vec_init as *const ())));
        }

        let conn = Connection::open_in_memory().context("Failed to open in memory database")?;

        // Create a regular table with BLOB for embeddings and CHECK constraints
        conn.execute(
            format!(
                "CREATE TABLE clips (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transcript_embedding BLOB CHECK(vec_length(transcript_embedding) = {}),
                    start_time FLOAT,
                    end_time FLOAT,
                    transcript TEXT
                );",
                dimensions
            )
            .as_str(),
            params![],
        )
        .context("Failed to create table")?;

        Ok(Self { conn })
    }

    pub fn add_clip<E: Embedder>(&self, embedder: &mut E, timestamp: &Timestamp) -> Result<()> {
        let transcript = timestamp.text.clone();
        let start_time = timestamp.start;
        let end_time = timestamp.end;

        let transcript_embedding = embedder.embed(&transcript)?;

        self.conn.execute(
            "INSERT INTO clips (transcript_embedding, start_time, end_time, transcript) VALUES (?, ?, ?, ?)",
            params![transcript_embedding.as_bytes(), start_time, end_time, transcript],
        )?;

        Ok(())
    }

    pub fn batch_add_clips<E: Embedder>(
        &self,
        embedder: &mut E,
        timestamps: Vec<Timestamp>,
    ) -> Result<()> {
        let transcripts = timestamps
            .iter()
            .map(|t| t.text.clone())
            .collect::<Vec<String>>();

        let transcript_embeddings = embedder.batch_embed(&transcripts)?;

        let clips: Vec<(Vec<f32>, f64, f64, String)> = timestamps
            .into_iter()
            .zip(transcript_embeddings.into_iter())
            .map(|(t, e)| (e, t.start, t.end, t.text))
            .collect();

        let mut stmt = self.conn.prepare(
            "INSERT INTO clips (transcript_embedding, start_time, end_time, transcript) VALUES (?, ?, ?, ?)",
        )?;

        for clip in clips {
            stmt.execute(params![clip.0.as_bytes(), clip.1, clip.2, clip.3])?;
        }

        Ok(())
    }

    pub fn search<E: Embedder>(
        &self,
        embedder: &mut E,
        query: &str,
        max_results: usize,
    ) -> Result<Vec<DBTimestamp>> {
        let query_embedding = embedder.embed(query)?;

        let mut stmt = self.conn.prepare(
            "SELECT id, start_time, end_time, transcript 
             FROM clips 
             ORDER BY vec_distance_cosine(transcript_embedding, ?) ASC 
             LIMIT ?",
        )?;

        let mut rows = stmt.query_map(params![query_embedding.as_bytes(), max_results], |row| {
            let id: i64 = row.get("id")?;
            let start_time: f64 = row.get("start_time")?;
            let end_time: f64 = row.get("end_time")?;
            let transcript: String = row.get("transcript")?;

            Ok(DBTimestamp {
                id,
                start_time,
                end_time,
                transcript,
            })
        })?;

        let mut results = Vec::new();
        while let Some(Ok(timestamp)) = rows.next() {
            results.push(timestamp);
        }

        Ok(results)
    }

    pub fn get_neighboring_clips(
        &self,
        clip_id: i64,
        num_neighbors_before: usize,
        num_neighbors_after: usize,
    ) -> Result<Vec<DBTimestamp>> {
        // Get the target clip
        let target_clip: DBTimestamp = self.conn.query_row(
            "SELECT id, start_time, end_time, transcript FROM clips WHERE id = ?",
            params![clip_id],
            |row| {
                Ok(DBTimestamp {
                    id: row.get("id")?,
                    start_time: row.get("start_time")?,
                    end_time: row.get("end_time")?,
                    transcript: row.get("transcript")?,
                })
            },
        )?;

        // Get clips before the target time
        let mut stmt = self.conn.prepare(
            "SELECT id, start_time, end_time, transcript 
             FROM clips 
             WHERE id != ? AND start_time < ?
             ORDER BY start_time DESC
             LIMIT ?",
        )?;

        let before_rows = stmt.query_map(
            params![clip_id, target_clip.start_time, num_neighbors_before],
            |row| {
                Ok(DBTimestamp {
                    id: row.get("id")?,
                    start_time: row.get("start_time")?,
                    end_time: row.get("end_time")?,
                    transcript: row.get("transcript")?,
                })
            },
        )?;

        // Get clips after the target time
        let mut stmt = self.conn.prepare(
            "SELECT id, start_time, end_time, transcript 
             FROM clips 
             WHERE id != ? AND start_time > ?
             ORDER BY start_time ASC
             LIMIT ?",
        )?;

        let after_rows = stmt.query_map(
            params![clip_id, target_clip.start_time, num_neighbors_after],
            |row| {
                Ok(DBTimestamp {
                    id: row.get("id")?,
                    start_time: row.get("start_time")?,
                    end_time: row.get("end_time")?,
                    transcript: row.get("transcript")?,
                })
            },
        )?;

        // Combine results in chronological order
        let mut results = Vec::new();

        // Add before clips in reverse order (to maintain chronological order)
        let mut before_clips: Vec<_> = before_rows.filter_map(Result::ok).collect();
        before_clips.reverse();
        results.extend(before_clips);

        // Add the target clip in the middle
        results.push(target_clip);

        // Add after clips
        results.extend(after_rows.filter_map(Result::ok));

        Ok(results)
    }

    pub fn get_clips_in_range(&self, start_time: f64, end_time: f64) -> Result<Vec<Timestamp>> {
        let mut stmt = self.conn.prepare(
            "SELECT start_time, end_time, transcript 
             FROM clips 
             WHERE (start_time BETWEEN ? AND ?) OR (end_time BETWEEN ? AND ?) 
             ORDER BY start_time ASC",
        )?;

        let mut rows =
            stmt.query_map(params![start_time, end_time, start_time, end_time], |row| {
                let start_time: f64 = row.get("start_time")?;
                let end_time: f64 = row.get("end_time")?;
                let transcript: String = row.get("transcript")?;

                Ok(Timestamp {
                    start: start_time,
                    end: end_time,
                    text: transcript,
                })
            })?;

        let mut results = Vec::new();
        while let Some(Ok(timestamp)) = rows.next() {
            results.push(timestamp);
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // Mock implementation of Llama for testing
    struct MockLlama {
        counter: AtomicUsize,
    }

    impl MockLlama {
        fn new() -> Self {
            Self {
                counter: AtomicUsize::new(0),
            }
        }

        fn generate_embedding(&self, _text: &str) -> Vec<f32> {
            let count = self.counter.fetch_add(1, Ordering::SeqCst);
            vec![count as f32; 384]
        }
    }

    impl Embedder for MockLlama {
        fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
            Ok(self.generate_embedding(text))
        }

        fn batch_embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|t| self.generate_embedding(t)).collect())
        }
    }

    fn create_test_timestamps() -> Vec<Timestamp> {
        vec![
            Timestamp {
                start: 0.0,
                end: 10.0,
                text: "Hello world".to_string(),
            },
            Timestamp {
                start: 10.0,
                end: 20.0,
                text: "This is a test".to_string(),
            },
            Timestamp {
                start: 20.0,
                end: 30.0,
                text: "Testing vector search".to_string(),
            },
        ]
    }

    #[test]
    fn test_vector_db_operations() -> Result<()> {
        let db = VectorDB::new_in_memory(384)?;
        let mut llama = MockLlama::new();

        let timestamp = Timestamp {
            start: 0.0,
            end: 10.0,
            text: "Hello world".to_string(),
        };
        db.add_clip(&mut llama, &timestamp)?;

        let timestamps = create_test_timestamps();
        db.batch_add_clips(&mut llama, timestamps.clone())?;

        let results = db.search(&mut llama, "test", 2)?;
        assert!(!results.is_empty(), "Search should return results");
        assert!(
            results.len() <= 2,
            "Search should respect max_results parameter"
        );

        Ok(())
    }

    #[test]
    fn test_vector_db_empty_search() -> Result<()> {
        let db = VectorDB::new_in_memory(384)?;
        let mut llama = MockLlama::new();

        let results = db.search(&mut llama, "test", 5)?;
        assert!(
            results.is_empty(),
            "Search on empty database should return no results"
        );

        Ok(())
    }

    #[test]
    fn test_vector_db_max_results() -> Result<()> {
        let db = VectorDB::new_in_memory(384)?;
        let mut llama = MockLlama::new();

        let timestamps = create_test_timestamps();
        db.batch_add_clips(&mut llama, timestamps)?;

        let results1 = db.search(&mut llama, "test", 1)?;
        assert_eq!(results1.len(), 1, "Should return exactly 1 result");

        let results2 = db.search(&mut llama, "test", 2)?;
        assert!(results2.len() <= 2, "Should return at most 2 results");

        Ok(())
    }

    #[test]
    fn test_get_neighboring_clips() -> Result<()> {
        let db = VectorDB::new_in_memory(384)?;
        let mut llama = MockLlama::new();

        let timestamps = create_test_timestamps();
        db.batch_add_clips(&mut llama, timestamps)?;

        // Get neighbors of the middle clip (id = 2)
        let neighbors = db.get_neighboring_clips(2, 1, 1)?;
        assert_eq!(neighbors.len(), 2, "Should return exactly 2 neighbors");

        Ok(())
    }

    #[test]
    fn test_get_clips_in_range() -> Result<()> {
        let db = VectorDB::new_in_memory(384)?;
        let mut llama = MockLlama::new();

        let timestamps = create_test_timestamps();
        db.batch_add_clips(&mut llama, timestamps)?;

        let clips = db.get_clips_in_range(5.0, 15.0)?;
        assert!(!clips.is_empty(), "Should return clips in the range");
        assert!(
            clips
                .iter()
                .all(|c| (c.start <= 15.0 && c.start >= 5.0) || (c.end <= 15.0 && c.end >= 5.0)),
            "All clips should overlap with the range"
        );

        Ok(())
    }
}
