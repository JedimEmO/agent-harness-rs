use diesel::prelude::*;
use diesel::r2d2::{self, ConnectionManager, Pool};
use diesel::connection::SimpleConnection;
use diesel_migrations::{embed_migrations, EmbeddedMigrations, MigrationHarness};
use std::path::Path;

use crate::error::DbError;

pub type DbPool = Pool<ConnectionManager<SqliteConnection>>;

const MIGRATIONS: EmbeddedMigrations = embed_migrations!("migrations");

#[derive(Debug)]
struct SqlitePragmas;

impl r2d2::CustomizeConnection<SqliteConnection, r2d2::Error> for SqlitePragmas {
    fn on_acquire(&self, conn: &mut SqliteConnection) -> Result<(), r2d2::Error> {
        conn.batch_execute("PRAGMA foreign_keys = ON")
            .expect("failed to enable foreign keys");
        Ok(())
    }
}

/// Initialize the SQLite database pool and run migrations.
pub fn init_db(db_path: &str) -> Result<DbPool, DbError> {
    if let Some(parent) = Path::new(db_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }

    let manager = ConnectionManager::<SqliteConnection>::new(db_path);
    let pool = Pool::builder()
        .connection_customizer(Box::new(SqlitePragmas))
        .build(manager)
        .map_err(|e| DbError::Connection(e.to_string()))?;

    let mut conn = pool.get().map_err(|e| DbError::Connection(e.to_string()))?;
    conn.batch_execute("PRAGMA journal_mode=WAL")
        .map_err(|e| DbError::Query(e.to_string()))?;

    conn.run_pending_migrations(MIGRATIONS)
        .map_err(|e| DbError::Query(e.to_string()))?;

    Ok(pool)
}

pub async fn with_conn<F, T>(pool: &DbPool, f: F) -> Result<T, DbError>
where
    F: FnOnce(&mut SqliteConnection) -> Result<T, diesel::result::Error> + Send + 'static,
    T: Send + 'static,
{
    let pool = pool.clone();
    tokio::task::spawn_blocking(move || {
        let mut conn = pool.get().map_err(|e| DbError::Connection(e.to_string()))?;
        f(&mut conn).map_err(|e| match e {
            diesel::result::Error::NotFound => DbError::NotFound,
            e => DbError::Query(e.to_string()),
        })
    })
    .await
    .map_err(|e| DbError::Query(e.to_string()))?
}
