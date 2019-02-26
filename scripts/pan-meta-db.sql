CREATE TABLE units (
	id integer PRIMARY KEY,
	name varchar(255),
	lon numeric(6,2),
	lat numeric(6,2),
	elevation numeric(7,2),
	commission_date date,
	active boolean DEFAULT true
);

CREATE TABLE cameras (
	id char(6) UNIQUE,
	unit_id integer REFERENCES units (id) ON DELETE RESTRICT NOT NULL,
	commission_date date,
	active boolean DEFAULT true
);

CREATE TABLE sequences (
	id char(29) PRIMARY KEY,
	unit_id integer REFERENCES units (id) ON DELETE RESTRICT NOT NULL,
	start_date timestamp UNIQUE,
  	coord_bounds box,
  	exptime numeric(6,2), 
  	pocs_version varchar(15),
  	piaa_state varchar(45) DEFAULT 'initial'

);
CREATE INDEX on sequences (start_date);
CREATE INDEX on sequences (piaa_state);

CREATE TABLE images (
	id char(29) PRIMARY KEY,
	sequence_id char(29) REFERENCES sequences (id) ON DELETE RESTRICT NOT NULL,
	camera_id char(6) REFERENCES cameras (id) ON DELETE RESTRICT NOT NULL,
	obstime timestamp,
	center_ra numeric(7,4),
	center_dec numeric(7,4),
	ra_mnt numeric(7,4), 
	ha_mnt numeric(7,4),
	dec_mnt numeric(7,4), 
	exptime numeric(6,2),
	headers jsonb,
  	file_path text
);
CREATE INDEX on images (file_path);
CREATE INDEX on images (center_ra);
CREATE INDEX on images (center_dec);
CREATE INDEX on images (obstime);

CREATE TABLE stamps (
	image_id char(29) REFERENCES images (id) NOT NULL,
	picid bigint NOT NULL,
	obstime timestamp,
	ra numeric(7,4), 
	dec numeric(7,4),	
	x_pos numeric(7,4),	
	y_pos numeric(7,4),	
	data float ARRAY,
	PRIMARY KEY (picid, image_id)
);
CREATE INDEX on stamps (picid);
CREATE INDEX on stamps (ra);
CREATE INDEX on stamps (dec);
