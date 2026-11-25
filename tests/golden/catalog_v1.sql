-- A catalog at schema version 1, as betautils_catalog_store wrote it,
-- with synthetic rows. The fixture for migrating a previous version.
-- Never regenerate it with newer code: it stands for files made by v1.
PRAGMA user_version = 1;
BEGIN TRANSACTION;
CREATE TABLE folders (
    id               INTEGER PRIMARY KEY,
    path             TEXT    NOT NULL UNIQUE,
    type_from_folder INTEGER NOT NULL DEFAULT 0,
    added_at         REAL    NOT NULL
);
INSERT INTO "folders" VALUES(1,'/library/source',1,1000.0);
CREATE TABLE items (
    hash             TEXT    PRIMARY KEY,
    kind             TEXT    NOT NULL DEFAULT 'video',
    size             INTEGER NOT NULL,
    name             TEXT    NOT NULL,
    added_at         REAL    NOT NULL,
    status           TEXT    NOT NULL DEFAULT 'new',
    type_pin         TEXT,
    last_profile     TEXT,
    last_rule        TEXT,
    last_rule_params TEXT
);
INSERT INTO "items" VALUES('0123456789abcdef','video',4096,'clip.mp4',1001.0,'done','quick_cut','scene','source_folder','{"folder": "scene"}');
INSERT INTO "items" VALUES('fedcba9876543210','video',2048,'other.mkv',1002.0,'new',NULL,NULL,NULL,NULL);
CREATE TABLE locations (
    id        INTEGER PRIMARY KEY,
    hash      TEXT    NOT NULL REFERENCES items( hash ) ON DELETE CASCADE,
    path      TEXT    NOT NULL UNIQUE,
    folder_id INTEGER REFERENCES folders( id ) ON DELETE SET NULL,
    size      INTEGER NOT NULL,
    mtime_ns  INTEGER NOT NULL,
    dev       INTEGER NOT NULL,
    ino       INTEGER NOT NULL,
    state     TEXT    NOT NULL DEFAULT 'present',
    added_at  REAL    NOT NULL,
    seen_at   REAL    NOT NULL
);
INSERT INTO "locations" VALUES(1,'0123456789abcdef','/library/source/scene/clip.mp4',1,4096,5,1,11,'present',1001.0,1003.0);
INSERT INTO "locations" VALUES(2,'fedcba9876543210','/library/source/other.mkv',1,2048,6,1,12,'missing',1002.0,1003.0);
CREATE TABLE outputs (
    id            INTEGER PRIMARY KEY,
    hash          TEXT    NOT NULL REFERENCES items( hash ) ON DELETE CASCADE,
    path          TEXT    NOT NULL UNIQUE,
    detection_key TEXT    NOT NULL,
    censor_key    TEXT    NOT NULL,
    encode_key    TEXT    NOT NULL,
    present       INTEGER NOT NULL DEFAULT 1,
    seen_at       REAL    NOT NULL
);
INSERT INTO "outputs" VALUES(1,'0123456789abcdef','/library/censored_vids/clip-0123456789abcdef-640-9-d1-c2-e3.mp4','d1','c2','e3',1,1003.0);
CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at REAL NOT NULL,
                app_version TEXT, checksum TEXT NOT NULL );
INSERT INTO "schema_migrations" VALUES(1,'items_locations_folders_outputs',1000.0,'1.5.2+12.g4d6aaef.dirty','2d41624e4d6d14ef75d5c8f4f1aff89e5c2533e676768b08cabd6c95b5005de2');
CREATE INDEX locations_hash  ON locations( hash );
CREATE INDEX locations_inode ON locations( dev, ino );
CREATE INDEX outputs_hash ON outputs( hash );
COMMIT;
