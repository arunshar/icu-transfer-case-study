-- pipeline.sql  --  SQL companion to extract.py (satisfies "extract.py OR pipeline.sql").
--
-- Expresses the cohort + 12-hour pre-transfer window selection in pure SQL against
-- the SQLite database that extract.py builds at data/mimic_demo.sqlite.
--
--   Build the DB then run this file:
--     python extract.py                          # writes data/mimic_demo.sqlite
--     sqlite3 data/mimic_demo.sqlite < pipeline.sql
--
-- Narrative synthesis and matched-control sampling stay in extract.py (procedural);
-- this script is the auditable SQL definition of the *positive* cohort and its window.

-- ---------------------------------------------------------------------------
-- 1) Target event: a ward -> ICU TRANSFER (not a direct ICU admission).
--    eventtype='transfer', curr_careunit is an ICU unit, prev_careunit is not.
--    The transfer timestamp is the ICU segment INTIME.
-- ---------------------------------------------------------------------------
WITH icu_units(u) AS (
    VALUES ('MICU'), ('SICU'), ('CCU'), ('CSRU'), ('TSICU'), ('NICU')
),
ward2icu AS (
    SELECT
        t.subject_id,
        t.hadm_id,
        t.intime AS event_time,
        ROW_NUMBER() OVER (PARTITION BY t.hadm_id ORDER BY t.intime) AS rn
    FROM transfers t
    WHERE t.eventtype = 'transfer'
      AND t.curr_careunit IN (SELECT u FROM icu_units)
      AND (t.prev_careunit IS NULL OR t.prev_careunit NOT IN (SELECT u FROM icu_units))
),
positives AS (                      -- first escalation per admission + its 12h window
    SELECT
        subject_id,
        hadm_id,
        event_time,
        datetime(event_time, '-13 hours') AS window_start,   -- 12h window ...
        datetime(event_time, '-1 hours')  AS window_end      -- ... ending 1h pre-transfer
    FROM ward2icu
    WHERE rn = 1
)

-- ---------------------------------------------------------------------------
-- 2) Pull the labs that fall strictly inside the pre-event window (temporal
--    integrity: charttime >= window_start AND charttime < window_end).
--    One row per (patient, lab) with the latest in-window value.
-- ---------------------------------------------------------------------------
SELECT
    p.subject_id,
    p.hadm_id,
    p.event_time,
    p.window_start,
    p.window_end,
    dl.label                         AS lab,
    le.valuenum                      AS latest_value,
    le.valueuom                      AS unit,
    le.charttime                     AS measured_at
FROM positives p
JOIN labevents le
    ON le.hadm_id = p.hadm_id
   AND le.charttime >= p.window_start
   AND le.charttime <  p.window_end
   AND le.valuenum IS NOT NULL
JOIN d_labitems dl
    ON dl.itemid = le.itemid
-- keep only the most recent in-window measurement per lab per patient
WHERE le.charttime = (
    SELECT MAX(le2.charttime)
    FROM labevents le2
    WHERE le2.hadm_id = le.hadm_id
      AND le2.itemid  = le.itemid
      AND le2.charttime >= p.window_start
      AND le2.charttime <  p.window_end
)
ORDER BY p.window_end, p.hadm_id, dl.label;
