CREATE TABLE IF NOT EXISTS units (
	id integer PRIMARY KEY,
	name varchar(255),
	lon numeric(6,2),
	lat numeric(6,2),
	elevation numeric(7,2),
	commission_date date,
	active boolean DEFAULT true
);

CREATE TABLE IF NOT EXISTS cameras (
	id char(6) UNIQUE,
	unit_id integer REFERENCES units (id) ON DELETE RESTRICT NOT NULL,
	commission_date date,
	active boolean DEFAULT true
);

CREATE TABLE IF NOT EXISTS sequences (
	id char(29) PRIMARY KEY,
	unit_id integer REFERENCES units (id) ON DELETE RESTRICT NOT NULL,
	start_date timestamp UNIQUE,
  	exptime numeric(6,2), 
  	pocs_version varchar(15),
  	field varchar(255),
  	state varchar(45) DEFAULT 'initial',
    dec_min numeric(7, 4),
    dec_max numeric(7, 4),
    ra_min numeric(7, 4),
    ra_max numeric(7, 4)
);
CREATE INDEX IF NOT EXISTS sequences_start_date_idex on sequences (start_date);
CREATE INDEX IF NOT EXISTS sequences_state_idx on sequences (state);
CREATE INDEX IF NOT EXISTS sequences_dec_min_dec_max_idx on sequences (dec_min, dec_max);
CREATE INDEX IF NOT EXISTS sequcnes_ra_min_ra_max_idx on sequences (ra_min, ra_max);

CREATE TABLE IF NOT EXISTS images (
	id char(29) PRIMARY KEY,
	sequence_id char(29) REFERENCES sequences (id) ON DELETE RESTRICT NOT NULL,
	camera_id char(6) REFERENCES cameras (id) ON DELETE RESTRICT NOT NULL,
	obstime timestamp,
	ra_mnt numeric(7,4), 
	ha_mnt numeric(7,4),
	dec_mnt numeric(7,4), 
	exptime numeric(6,2),
	headers jsonb,
  	file_path text,
  	state varchar(45) DEFAULT 'initial'
);
CREATE INDEX IF NOT EXISTS images_file_path_idx on images (file_path);
CREATE INDEX IF NOT EXISTS images_obstime_idx on images (obstime);
CREATE INDEX IF NOT EXISTS images_sequence_id_idx on images (sequence_id);

CREATE TABLE IF NOT EXISTS stamps (
	picid bigint NOT NULL,
	image_id char(29) REFERENCES images (id) NOT NULL,
	obstime timestamp,
    astro_coords point,
    pixel_coords point,
	data float ARRAY,
    metadata jsonb,
	PRIMARY KEY (picid, image_id)
);
CREATE INDEX IF NOT EXISTS stamps_picid_idx on stamps (picid);
CREATE INDEX IF NOT EXISTS stamps_image_id_idx on stamps (image_id);
CREATE INDEX IF NOT EXISTS stamps_astro_coords_idx on stamps USING SPGIST (astro_coords);

CREATE EXTENSION IF NOT EXISTS intarray;

DROP FUNCTION normalize_stamp(float[]);
CREATE OR REPLACE FUNCTION normalize_stamp(d0 float[]) RETURNS SETOF float[] AS $$
    SELECT array_agg(norms.norm) FROM
    (
        SELECT (
                UNNEST(d0) / (SELECT stamp_sum FROM (SELECT SUM(d1) as stamp_sum FROM UNNEST(d0) as d1) as t0)
                ) as norm
    ) as norms
$$ LANGUAGE SQL;

/*
CREATE OR REPLACE VIEW stamp_data AS
    SELECT 
        t2.sequence_id,
        t1.image_id,
        t1.picid,
        t1.obstime,
        t1.data, 
        array_agg(t1.n0) AS normalized 
    FROM 
        (
            SELECT *, d0/sum(d0) over (partition by obstime) n0 
            FROM (
                    SELECT *, unnest(data) AS d0
                    FROM stamps 
                 ) AS t0
        ) AS t1, 
        images AS t2
    WHERE t2.id=t1.image_id 
    GROUP BY t2.sequence_id, t1.picid, t1.obstime, t1.data, t1.image_id 
    ORDER BY t1.obstime;
*/