CREATE TABLE units (
	id integer PRIMARY KEY,
	name varchar(255),
	lon numeric(6,2),
	lat numeric(6,2),
	elevation numeric(7,2),
	commission_date date,
	active boolean DEFAULT false
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
	-- ra numeric(7,4),
	-- dec numeric(7,4),
	start_date timestamp UNIQUE,
  	-- set_size smallint,
  	-- priority numeric(6,2),
  	exp_time numeric(6,2), 
  	ra_rate numeric(3,2),  
  	-- merit numeric(6,2), 
  	-- min_nexp smallint, 
  	-- min_duration numeric(7,2), 
  	-- set_duration numeric(7,2),
  	pocs_version varchar(15)
);
-- CREATE INDEX on sequences (ra);
-- CREATE INDEX on sequences (dec);
CREATE INDEX on sequences (start_date);

CREATE TABLE images (
	id char(29) PRIMARY KEY,
	sequence_id char(29) REFERENCES sequences (id) ON DELETE RESTRICT NOT NULL,
	camera_id char(6) REFERENCES cameras (id) ON DELETE RESTRICT NOT NULL,
	date_obs timestamp,
	center_ra numeric(7,4),
	center_dec numeric(7,4),
	moon_fraction numeric(4,3),
	moon_separation numeric(6,3),
	ra_mnt numeric(7,4), 
	ha_mnt numeric(7,4),
	dec_mnt numeric(7,4), 
	airmass numeric(4,3),
	exp_time numeric(6,2),
	iso int,
	cam_temp numeric(4,2), 
	cam_circconf numeric(4,3),
	cam_colortmp numeric(4,0),
  cam_measrggb varchar(20),
  cam_measured_ev numeric(4,2),
	cam_measured_ev2 numeric(4,2),
  cam_white_level_n int,
  cam_white_level_s int,
	cam_red_balance numeric(7,6),
	cam_blue_balance numeric(7,6),
  file_path text
);
CREATE INDEX on images (file_path);
CREATE INDEX on images (center_ra);
CREATE INDEX on images (center_dec);