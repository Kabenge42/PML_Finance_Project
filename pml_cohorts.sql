-- Basic retention
SELECT isin, min(term_start) AS first_term
FROM legislators_terms
GROUP BY 1
;

SELECT date_part('year', age(b.term_start, a.first_term)) AS periods, count(DISTINCT a.isin) AS cohort_retained
FROM (SELECT isin, min(term_start) AS first_term FROM legislators_terms GROUP BY 1
     )                          a
	     JOIN legislators_terms b ON a.isin = b.isin
GROUP BY 1
;

SELECT period
	 , first_value(cohort_retained) OVER (ORDER BY period)                         AS cohort_size
	 , cohort_retained
	 , cohort_retained * 1.0 / first_value(cohort_retained) OVER (ORDER BY period) AS pct_retained
FROM (SELECT date_part('year', age(b.term_start, a.first_term)) AS period, count(DISTINCT a.isin) AS cohort_retained
      FROM (SELECT isin, min(term_start) AS first_term FROM legislators_terms GROUP BY 1
           )                          a
	           JOIN legislators_terms b ON a.isin = b.isin
      GROUP BY 1
     ) aa
;

SELECT cohort_size
	 , max(CASE WHEN period = 0 THEN pct_retained END) AS yr0
	 , max(CASE WHEN period = 1 THEN pct_retained END) AS yr1
	 , max(CASE WHEN period = 2 THEN pct_retained END) AS yr2
	 , max(CASE WHEN period = 3 THEN pct_retained END) AS yr3
	 , max(CASE WHEN period = 4 THEN pct_retained END) AS yr4
FROM (SELECT period
           , first_value(cohort_retained) OVER (ORDER BY period)                         AS cohort_size
           , cohort_retained
           , cohort_retained * 1.0 / first_value(cohort_retained) OVER (ORDER BY period) AS pct_retained
      FROM (SELECT date_part('year', age(b.term_start, a.first_term)) AS period, count(*) AS cohort_retained
            FROM (SELECT isin, min(term_start) AS first_term FROM legislators_terms GROUP BY 1
                 )                          a
	                 JOIN legislators_terms b ON a.isin = b.isin
            GROUP BY 1
           ) aa
     ) aaa
GROUP BY 1
;

-- Time adjustments
SELECT a.isin
	 , a.first_term
	 , b.term_start
	 , b.term_end
	 , c.date
	 , date_part('year', age(c.date, a.first_term)) AS period
FROM (SELECT isin, min(term_start) AS first_term FROM legislators_terms GROUP BY 1
     )                               a
	     JOIN      legislators_terms b ON a.isin = b.isin
	     LEFT JOIN date_dim          c
	               ON c.date BETWEEN b.term_start AND b.term_end AND c.month_name = 'December' AND c.day_of_month = 31
;

SELECT coalesce(date_part('year', age(c.date, a.first_term)), 0) AS period
	 , count(DISTINCT a.isin)                                    AS cohort_retained
FROM (SELECT isin, min(term_start) AS first_term FROM legislators_terms GROUP BY 1
     )                               a
	     JOIN      legislators_terms b ON a.isin = b.isin
	     LEFT JOIN date_dim          c
	               ON c.date BETWEEN b.term_start AND b.term_end AND c.month_name = 'December' AND c.day_of_month = 31
GROUP BY 1
;

SELECT period
	 , first_value(cohort_retained) OVER (ORDER BY period)                         AS cohort_size
	 , cohort_retained
	 , cohort_retained * 1.0 / first_value(cohort_retained) OVER (ORDER BY period) AS pct_retained
FROM (SELECT coalesce(date_part('year', age(c.date, a.first_term)), 0) AS period
           , count(DISTINCT a.isin)                                    AS cohort_retained
      FROM (SELECT isin, min(term_start) AS first_term FROM legislators_terms GROUP BY 1
           )                               a
	           JOIN      legislators_terms b ON a.isin = b.isin
	           LEFT JOIN date_dim          c
	                     ON c.date BETWEEN b.term_start AND b.term_end AND c.month_name = 'December' AND
	                        c.day_of_month = 31
      GROUP BY 1
     ) aa
;

SELECT a.isin
	 , a.first_term
	 , b.term_start
	 , CASE
		   WHEN b.term_type = 'rep' THEN b.term_start + INTERVAL '2 years'
		   WHEN b.term_type = 'sen' THEN b.term_start + INTERVAL '6 years' END AS term_end
FROM (SELECT isin, min(term_start) AS first_term FROM legislators_terms GROUP BY 1
     )                          a
	     JOIN legislators_terms b ON a.isin = b.isin
;

SELECT a.isin
	 , a.first_term
	 , b.term_start
	 , lead(b.term_start) OVER (PARTITION BY a.isin ORDER BY b.term_start) - INTERVAL '1 day' AS term_end
FROM (SELECT isin, min(term_start) AS first_term FROM legislators_terms GROUP BY 1
     )                          a
	     JOIN legislators_terms b ON a.isin = b.isin
ORDER BY 1, 3
;

-- Time-based cohorts derived from the time-series

SELECT date_part('year', a.first_term)                           AS first_year
	 , coalesce(date_part('year', age(c.date, a.first_term)), 0) AS period
	 , count(DISTINCT a.isin)                                    AS cohort_retained
FROM (SELECT isin, min(term_start) AS first_term FROM legislators_terms GROUP BY 1
     )                               a
	     JOIN      legislators_terms b ON a.isin = b.isin
	     LEFT JOIN date_dim          c
	               ON c.date BETWEEN b.term_start AND b.term_end AND c.month_name = 'December' AND c.day_of_month = 31
GROUP BY 1, 2
;

SELECT first_year
	 , period
	 , first_value(cohort_retained) OVER (PARTITION BY first_year ORDER BY period)                                   AS cohort_size
	 , cohort_retained
	 , round(cohort_retained * 1.0 / first_value(cohort_retained) OVER (PARTITION BY first_year ORDER BY period),
	         2)                                                                                                      AS pct_retained
FROM (SELECT date_part('year', first_term)                      AS first_year
           , date_part('year', age(b.term_start, a.first_term)) AS period
           , count(DISTINCT a.isin)                             AS cohort_retained
      FROM (SELECT isin, min(term_start) AS first_term FROM legislators_terms GROUP BY 1
           )                          a
	           JOIN legislators_terms b ON a.isin = b.isin
      GROUP BY 1, 2
     ) aa
;

SELECT first_century
	 , period
	 , first_value(cohort_retained) OVER (PARTITION BY first_century ORDER BY period)                         AS cohort_size
	 , cohort_retained
	 , cohort_retained * 1.0 / first_value(cohort_retained)
	                           OVER (PARTITION BY first_century ORDER BY period)                              AS pct_retained
FROM (SELECT date_part('century', a.first_term)                        AS first_century
           , coalesce(date_part('year', age(c.date, a.first_term)), 0) AS period
           , count(DISTINCT a.isin)                                    AS cohort_retained
      FROM (SELECT isin, min(term_start) AS first_term FROM legislators_terms GROUP BY 1
           )                               a
	           JOIN      legislators_terms b ON a.isin = b.isin
	           LEFT JOIN date_dim          c
	                     ON c.date BETWEEN b.term_start AND b.term_end AND c.month_name = 'December' AND
	                        c.day_of_month = 31
      GROUP BY 1, 2
     ) aa
ORDER BY 1, 2
;

SELECT DISTINCT
       isin
	 , min(term_start) OVER (PARTITION BY isin)                        AS first_term
	 , first_value(state) OVER (PARTITION BY isin ORDER BY term_start) AS first_state
FROM legislators_terms
;

SELECT first_state
	 , period
	 , first_value(cohort_retained) OVER (PARTITION BY first_state ORDER BY period)                         AS cohort_size
	 , cohort_retained
	 , cohort_retained * 1.0 / first_value(cohort_retained)
	                           OVER (PARTITION BY first_state ORDER BY period)                              AS pct_retained
FROM (SELECT a.first_state
           , coalesce(date_part('year', age(c.date, a.first_term)), 0) AS period
           , count(DISTINCT a.isin)                                    AS cohort_retained
      FROM (SELECT DISTINCT
                   isin
	             , min(term_start) OVER (PARTITION BY isin)                        AS first_term
	             , first_value(state) OVER (PARTITION BY isin ORDER BY term_start) AS first_state
            FROM legislators_terms
           )                               a
	           JOIN      legislators_terms b ON a.isin = b.isin
	           LEFT JOIN date_dim          c
	                     ON c.date BETWEEN b.term_start AND b.term_end AND c.month_name = 'December' AND
	                        c.day_of_month = 31
      GROUP BY 1, 2
     ) aa
ORDER BY 1, 2
;

-- Defining the cohort from a separate table
SELECT d.gender
	 , coalesce(date_part('year', age(c.date, a.first_term)), 0) AS period
	 , count(DISTINCT a.isin)                                    AS cohort_retained
FROM (SELECT isin, min(term_start) AS first_term FROM legislators_terms GROUP BY 1
     )                               a
	     JOIN      legislators_terms b ON a.isin = b.isin
	     LEFT JOIN date_dim          c
	               ON c.date BETWEEN b.term_start AND b.term_end AND c.month_name = 'December' AND c.day_of_month = 31
	     JOIN      legislators       d ON a.isin = d.isin
GROUP BY 1, 2
ORDER BY 2, 1
;

SELECT gender
	 , period
	 , first_value(cohort_retained) OVER (PARTITION BY gender ORDER BY period)                         AS cohort_size
	 , cohort_retained
	 , cohort_retained * 1.0 / first_value(cohort_retained) OVER (PARTITION BY gender ORDER BY period) AS pct_retained
FROM (SELECT d.gender
           , coalesce(date_part('year', age(c.date, a.first_term)), 0) AS period
           , count(DISTINCT a.isin)                                    AS cohort_retained
      FROM (SELECT isin, min(term_start) AS first_term FROM legislators_terms GROUP BY 1
           )                               a
	           JOIN      legislators_terms b ON a.isin = b.isin
	           LEFT JOIN date_dim          c
	                     ON c.date BETWEEN b.term_start AND b.term_end AND c.month_name = 'December' AND
	                        c.day_of_month = 31
	           JOIN      legislators       d ON a.isin = d.isin
      GROUP BY 1, 2
     ) aa
ORDER BY 2, 1
;

SELECT gender
	 , period
	 , first_value(cohort_retained) OVER (PARTITION BY gender ORDER BY period)                         AS cohort_size
	 , cohort_retained
	 , cohort_retained * 1.0 / first_value(cohort_retained) OVER (PARTITION BY gender ORDER BY period) AS pct_retained
FROM (SELECT d.gender
           , coalesce(date_part('year', age(c.date, a.first_term)), 0) AS period
           , count(DISTINCT a.isin)                                    AS cohort_retained
      FROM (SELECT isin, min(term_start) AS first_term FROM legislators_terms GROUP BY 1
           )                               a
	           JOIN      legislators_terms b ON a.isin = b.isin
	           LEFT JOIN date_dim          c
	                     ON c.date BETWEEN b.term_start AND b.term_end AND c.month_name = 'December' AND
	                        c.day_of_month = 31
	           JOIN      legislators       d ON a.isin = d.isin
      WHERE a.first_term BETWEEN '1917-01-01' AND '1999-12-31'
      GROUP BY 1, 2
     ) aa
ORDER BY 2, 1
;

----------- Dealing with sparse cohorts
SELECT first_state
	 , gender
	 , period
	 , first_value(cohort_retained) OVER (PARTITION BY first_state, gender ORDER BY period)                   AS cohort_size
	 , cohort_retained
	 , cohort_retained / first_value(cohort_retained)
	                     OVER (PARTITION BY first_state, gender ORDER BY period)                              AS pct_retained
FROM (SELECT a.first_state
           , d.gender
           , coalesce(date_part('year', age(c.date, a.first_term)), 0) AS period
           , count(DISTINCT a.isin)                                    AS cohort_retained
      FROM (SELECT DISTINCT
                   isin
	             , min(term_start) OVER (PARTITION BY isin)                        AS first_term
	             , first_value(state) OVER (PARTITION BY isin ORDER BY term_start) AS first_state
            FROM legislators_terms
           )                               a
	           JOIN      legislators_terms b ON a.isin = b.isin
	           LEFT JOIN date_dim          c
	                     ON c.date BETWEEN b.term_start AND b.term_end AND c.month_name = 'December' AND
	                        c.day_of_month = 31
	           JOIN      legislators       d ON a.isin = d.isin
      WHERE a.first_term BETWEEN '1917-01-01' AND '1999-12-31'
      GROUP BY 1, 2, 3
     ) aa
;



SELECT aa.gender, aa.first_state, cc.period, aa.cohort_size
FROM (SELECT b.gender, a.first_state, count(DISTINCT a.isin) AS cohort_size
      FROM (SELECT DISTINCT
                   isin
	             , min(term_start) OVER (PARTITION BY isin)                        AS first_term
	             , first_value(state) OVER (PARTITION BY isin ORDER BY term_start) AS first_state
            FROM legislators_terms
           )                    a
	           JOIN legislators b ON a.isin = b.isin
      WHERE a.first_term BETWEEN '1917-01-01' AND '1999-12-31'
      GROUP BY 1, 2
     )              aa
	     JOIN (SELECT generate_series AS period FROM generate_series(0, 20, 1)
		          ) cc ON 1 = 1
;


SELECT aaa.gender
	 , aaa.first_state
	 , aaa.period
	 , aaa.cohort_size
	 , coalesce(ddd.cohort_retained, 0)                         AS cohort_retained
	 , coalesce(ddd.cohort_retained, 0) * 1.0 / aaa.cohort_size AS pct_retained
FROM (SELECT aa.gender, aa.first_state, cc.period, aa.cohort_size
      FROM (SELECT b.gender, a.first_state, count(DISTINCT a.isin) AS cohort_size
            FROM (SELECT DISTINCT
                         isin
	                   , min(term_start) OVER (PARTITION BY isin)                        AS first_term
	                   , first_value(state) OVER (PARTITION BY isin ORDER BY term_start) AS first_state
                  FROM legislators_terms
                 )                    a
	                 JOIN legislators b ON a.isin = b.isin
            WHERE a.first_term BETWEEN '1917-01-01' AND '1999-12-31'
            GROUP BY 1, 2
           )              aa
	           JOIN (SELECT generate_series AS period FROM generate_series(0, 20, 1)
		                ) cc ON 1 = 1
     )               aaa
	     LEFT JOIN (SELECT d.first_state
	                     , g.gender
	                     , coalesce(date_part('year', age(f.date, d.first_term)), 0) AS period
	                     , count(DISTINCT d.isin)                                    AS cohort_retained
	                FROM (SELECT DISTINCT
	                             isin
		                       , min(term_start) OVER (PARTITION BY isin)                        AS first_term
		                       , first_value(state) OVER (PARTITION BY isin ORDER BY term_start) AS first_state
	                      FROM legislators_terms
	                     )                               d
		                     JOIN      legislators_terms e ON d.isin = e.isin
		                     LEFT JOIN date_dim          f
		                               ON f.date BETWEEN e.term_start AND e.term_end AND f.month_name = 'December' AND
		                                  f.day_of_month = 31
		                     JOIN      legislators       g ON d.isin = g.isin
	                WHERE d.first_term BETWEEN '1917-01-01' AND '1999-12-31'
	                GROUP BY 1, 2, 3
	               ) ddd ON aaa.gender = ddd.gender AND aaa.first_state = ddd.first_state AND aaa.period = ddd.period
ORDER BY 1, 2, 3
;

SELECT gender
	 , first_state
	 , cohort_size
	 , max(CASE WHEN period = 0 THEN pct_retained END)  AS yr0
	 , max(CASE WHEN period = 2 THEN pct_retained END)  AS yr2
	 , max(CASE WHEN period = 4 THEN pct_retained END)  AS yr4
	 , max(CASE WHEN period = 6 THEN pct_retained END)  AS yr6
	 , max(CASE WHEN period = 8 THEN pct_retained END)  AS yr8
	 , max(CASE WHEN period = 10 THEN pct_retained END) AS yr10
FROM (SELECT aaa.gender
           , aaa.first_state
           , aaa.period
           , aaa.cohort_size
           , coalesce(ddd.cohort_retained, 0)                         AS cohort_retained
           , coalesce(ddd.cohort_retained, 0) * 1.0 / aaa.cohort_size AS pct_retained
      FROM (SELECT aa.gender, aa.first_state, cc.period, aa.cohort_size
            FROM (SELECT b.gender, a.first_state, count(DISTINCT a.isin) AS cohort_size
                  FROM (SELECT DISTINCT
                               isin
	                         , min(term_start) OVER (PARTITION BY isin)                        AS first_term
	                         , first_value(state) OVER (PARTITION BY isin ORDER BY term_start) AS first_state
                        FROM legislators_terms
                       )                    a
	                       JOIN legislators b ON a.isin = b.isin
                  WHERE a.first_term BETWEEN '1917-01-01' AND '1999-12-31'
                  GROUP BY 1, 2
                 )              aa
	                 JOIN (SELECT generate_series AS period FROM generate_series(0, 20, 1)
		                      ) cc ON 1 = 1
           )               aaa
	           LEFT JOIN (SELECT d.first_state
	                           , g.gender
	                           , coalesce(date_part('year', age(f.date, d.first_term)), 0) AS period
	                           , count(DISTINCT d.isin)                                    AS cohort_retained
	                      FROM (SELECT DISTINCT
	                                   isin
		                             , min(term_start) OVER (PARTITION BY isin)                        AS first_term
		                             , first_value(state) OVER (PARTITION BY isin ORDER BY term_start) AS first_state
	                            FROM legislators_terms
	                           )                               d
		                           JOIN      legislators_terms e ON d.isin = e.isin
		                           LEFT JOIN date_dim          f ON f.date BETWEEN e.term_start AND e.term_end AND
		                                                            f.month_name = 'December' AND f.day_of_month = 31
		                           JOIN      legislators       g ON d.isin = g.isin
	                      WHERE d.first_term BETWEEN '1917-01-01' AND '1999-12-31'
	                      GROUP BY 1, 2, 3
	                     ) ddd
	                     ON aaa.gender = ddd.gender AND aaa.first_state = ddd.first_state AND aaa.period = ddd.period
     ) a
GROUP BY 1, 2, 3
;

----------- Defining cohorts from dates other than the first date ----------------------------------

SELECT DISTINCT isin, term_type, date('2000-01-01') AS first_term, min(term_start) AS min_start
FROM legislators_terms
WHERE term_start <= '2000-12-31'
  AND term_end >= '2000-01-01'
GROUP BY 1, 2, 3
;


SELECT term_type
	 , period
	 , first_value(cohort_retained) OVER (PARTITION BY term_type ORDER BY period)                         AS cohort_size
	 , cohort_retained
	 , cohort_retained * 1.0 / first_value(cohort_retained)
	                           OVER (PARTITION BY term_type ORDER BY period)                              AS pct_retained
FROM (SELECT a.term_type
           , coalesce(date_part('year', age(c.date, a.first_term)), 0) AS period
           , count(DISTINCT a.isin)                                    AS cohort_retained
      FROM (SELECT DISTINCT isin, term_type, date('2000-01-01') AS first_term
            FROM legislators_terms
            WHERE term_start <= '2000-12-31'
	          AND term_end >= '2000-01-01'
           )                               a
	           JOIN      legislators_terms b ON a.isin = b.isin --and b.term_start >= a.first_term
	           LEFT JOIN date_dim          c
	                     ON c.date BETWEEN b.term_start AND b.term_end AND c.month_name = 'December' AND
	                        c.day_of_month = 31
      GROUP BY 1, 2
     ) aa
;

----------- Survivorship ----------------------------------
SELECT isin, min(term_start) AS first_term, max(term_start) AS last_term
FROM legislators_terms
GROUP BY 1
;


SELECT isin
	 , date_part('century', min(term_start))                    AS first_century
	 , min(term_start)                                          AS first_term
	 , max(term_start)                                          AS last_term
	 , date_part('year', age(max(term_start), min(term_start))) AS tenure
FROM legislators_terms
GROUP BY 1
;


SELECT first_century
	 , count(DISTINCT isin)                                                              AS cohort_size
	 , count(DISTINCT CASE WHEN tenure >= 10 THEN isin END)                              AS survived_10
	 , count(DISTINCT CASE WHEN tenure >= 10 THEN isin END) * 1.0 / count(DISTINCT isin) AS pct_survived_10
FROM (SELECT isin
           , date_part('century', min(term_start))                    AS first_century
           , min(term_start)                                          AS first_term
           , max(term_start)                                          AS last_term
           , date_part('year', age(max(term_start), min(term_start))) AS tenure
      FROM legislators_terms
      GROUP BY 1
     ) a
GROUP BY 1
;

SELECT isin
	 , date_part('century', min(term_start))                    AS first_century
	 , min(term_start)                                          AS first_term
	 , max(term_start)                                          AS last_term
	 , date_part('year', age(max(term_start), min(term_start))) AS tenure
FROM legislators_terms
GROUP BY 1
;


SELECT first_century
	 , count(DISTINCT isin)                                                                  AS cohort_size
	 , count(DISTINCT CASE WHEN total_terms >= 5 THEN isin END)                              AS survived_5
	 , count(DISTINCT CASE WHEN total_terms >= 5 THEN isin END) * 1.0 / count(DISTINCT isin) AS pct_survived_5_terms
FROM (SELECT isin, date_part('century', min(term_start)) AS first_century, count(term_start) AS total_terms
      FROM legislators_terms
      GROUP BY 1
     ) a
GROUP BY 1
;


SELECT a.first_century
	 , b.terms
	 , count(DISTINCT isin)                                                                          AS cohort
	 , count(DISTINCT CASE WHEN a.total_terms >= b.terms THEN isin END)                              AS cohort_survived
	 , count(DISTINCT CASE WHEN a.total_terms >= b.terms THEN isin END) * 1.0 / count(DISTINCT isin) AS pct_survived
FROM (SELECT isin, date_part('century', min(term_start)) AS first_century, count(term_start) AS total_terms
      FROM legislators_terms
      GROUP BY 1
     )              a
	     JOIN (SELECT generate_series AS terms FROM generate_series(1, 20, 1)
		          ) b ON 1 = 1
GROUP BY 1, 2
;

----------- Returnship / repeat purchase behavior ----------------------------------
SELECT date_part('century', a.first_term)::int AS cohort_century, count(isin) AS reps
FROM (SELECT isin, min(term_start) AS first_term FROM legislators_terms WHERE term_type = 'rep' GROUP BY 1
     ) a
GROUP BY 1
;

SELECT date_part('century', a.first_term) AS cohort_century, count(isin) AS reps
FROM (SELECT isin, min(term_start) AS first_term FROM legislators_terms WHERE term_type = 'rep' GROUP BY 1
     ) a
GROUP BY 1
ORDER BY 1
;

SELECT aa.cohort_century, bb.rep_and_sen * 1.0 / aa.reps AS pct_rep_and_sen
FROM (SELECT date_part('century', a.first_term) AS cohort_century, count(isin) AS reps
      FROM (SELECT isin, min(term_start) AS first_term FROM legislators_terms WHERE term_type = 'rep' GROUP BY 1
           ) a
      GROUP BY 1
     )               aa
	     LEFT JOIN (SELECT date_part('century', b.first_term) AS cohort_century, count(DISTINCT b.isin) AS rep_and_sen
	                FROM (SELECT isin, min(term_start) AS first_term
	                      FROM legislators_terms
	                      WHERE term_type = 'rep'
	                      GROUP BY 1
	                     )                          b
		                     JOIN legislators_terms c
		                          ON b.isin = c.isin AND c.term_type = 'sen' AND c.term_start > b.first_term
	                GROUP BY 1
	               ) bb ON aa.cohort_century = bb.cohort_century
;

SELECT aa.cohort_century, bb.rep_and_sen * 1.0 / aa.reps AS pct_rep_and_sen
FROM (SELECT date_part('century', a.first_term) AS cohort_century, count(isin) AS reps
      FROM (SELECT isin, min(term_start) AS first_term FROM legislators_terms WHERE term_type = 'rep' GROUP BY 1
           ) a
      WHERE first_term <= '2009-12-31'
      GROUP BY 1
     )               aa
	     LEFT JOIN (SELECT date_part('century', b.first_term) AS cohort_century, count(DISTINCT b.isin) AS rep_and_sen
	                FROM (SELECT isin, min(term_start) AS first_term
	                      FROM legislators_terms
	                      WHERE term_type = 'rep'
	                      GROUP BY 1
	                     )                          b
		                     JOIN legislators_terms c
		                          ON b.isin = c.isin AND c.term_type = 'sen' AND c.term_start > b.first_term
	                WHERE age(c.term_start, b.first_term) <= INTERVAL '10 years'
	                GROUP BY 1
	               ) bb ON aa.cohort_century = bb.cohort_century
;


SELECT aa.cohort_century::int                          AS cohort_century
	 , round(bb.rep_and_sen_5_yrs * 1.0 / aa.reps, 4)  AS pct_5_yrs
	 , round(bb.rep_and_sen_10_yrs * 1.0 / aa.reps, 4) AS pct_10_yrs
	 , round(bb.rep_and_sen_15_yrs * 1.0 / aa.reps, 4) AS pct_15_yrs
FROM (SELECT date_part('century', a.first_term) AS cohort_century, count(isin) AS reps
      FROM (SELECT isin, min(term_start) AS first_term FROM legislators_terms WHERE term_type = 'rep' GROUP BY 1
           ) a
      WHERE first_term <= '2009-12-31'
      GROUP BY 1
     )               aa
	     LEFT JOIN (SELECT date_part('century', b.first_term)      AS cohort_century
	                     , count(DISTINCT CASE
		                                      WHEN age(c.term_start, b.first_term) <= INTERVAL '5 years'
			                                      THEN b.isin END) AS rep_and_sen_5_yrs
	                     , count(DISTINCT CASE
		                                      WHEN age(c.term_start, b.first_term) <= INTERVAL '10 years'
			                                      THEN b.isin END) AS rep_and_sen_10_yrs
	                     , count(DISTINCT CASE
		                                      WHEN age(c.term_start, b.first_term) <= INTERVAL '15 years'
			                                      THEN b.isin END) AS rep_and_sen_15_yrs
	                FROM (SELECT isin, min(term_start) AS first_term
	                      FROM legislators_terms
	                      WHERE term_type = 'rep'
	                      GROUP BY 1
	                     )                          b
		                     JOIN legislators_terms c
		                          ON b.isin = c.isin AND c.term_type = 'sen' AND c.term_start > b.first_term
	                GROUP BY 1
	               ) bb ON aa.cohort_century = bb.cohort_century
;

----------- Cumulative calculations ----------------------------------
SELECT date_part('century', a.first_term)::int AS century
	 , first_type
	 , count(DISTINCT a.isin)                  AS cohort
	 , count(b.term_start)                     AS terms
FROM (SELECT DISTINCT
             isin
	       , first_value(term_type) OVER (PARTITION BY isin ORDER BY term_start) AS first_type
	       , min(term_start) OVER (PARTITION BY isin)                            AS first_term
	       , min(term_start) OVER (PARTITION BY isin) + INTERVAL '10 years'      AS first_plus_10
      FROM legislators_terms
     )                               a
	     LEFT JOIN legislators_terms b ON a.isin = b.isin AND b.term_start BETWEEN a.first_term AND a.first_plus_10
GROUP BY 1, 2
;

SELECT century
	 , max(CASE WHEN first_type = 'rep' THEN cohort END)        AS rep_cohort
	 , max(CASE WHEN first_type = 'rep' THEN terms_per_leg END) AS avg_rep_terms
	 , max(CASE WHEN first_type = 'sen' THEN cohort END)        AS sen_cohort
	 , max(CASE WHEN first_type = 'sen' THEN terms_per_leg END) AS avg_sen_terms
FROM (SELECT date_part('century', a.first_term)::int            AS century
           , first_type
           , count(DISTINCT a.isin)                             AS cohort
           , count(b.term_start)                                AS terms
           , count(b.term_start) * 1.0 / count(DISTINCT a.isin) AS terms_per_leg
      FROM (SELECT DISTINCT
                   isin
	             , first_value(term_type) OVER (PARTITION BY isin ORDER BY term_start) AS first_type
	             , min(term_start) OVER (PARTITION BY isin)                            AS first_term
	             , min(term_start) OVER (PARTITION BY isin) + INTERVAL '10 years'      AS first_plus_10
            FROM legislators_terms
           )                               a
	           LEFT JOIN legislators_terms b
	                     ON a.isin = b.isin AND b.term_start BETWEEN a.first_term AND a.first_plus_10
      GROUP BY 1, 2
     ) aa
GROUP BY 1
;

----------- Cross-section analysis, with a cohort lens ----------------------------------
SELECT b.date, count(DISTINCT a.isin) AS legislators
FROM legislators_terms a
	     JOIN date_dim b
	          ON b.date BETWEEN a.term_start AND a.term_end AND b.month_name = 'December' AND b.day_of_month = 31 AND
	             b.year <= 2019
GROUP BY 1
;

SELECT b.date, date_part('century', first_term)::int AS century, count(DISTINCT a.isin) AS legislators
FROM legislators_terms a
	     JOIN date_dim b
	          ON b.date BETWEEN a.term_start AND a.term_end AND b.month_name = 'December' AND b.day_of_month = 31 AND
	             b.year <= 2019
	     JOIN (SELECT isin, min(term_start) AS first_term FROM legislators_terms GROUP BY 1
	          )        c ON a.isin = c.isin
GROUP BY 1, 2
;

SELECT date
	 , century
	 , legislators
	 , sum(legislators) OVER (PARTITION BY date)                       AS cohort
	 , legislators * 100.0 / sum(legislators) OVER (PARTITION BY date) AS pct_century
FROM (SELECT b.date, date_part('century', first_term)::int AS century, count(DISTINCT a.isin) AS legislators
      FROM legislators_terms a
	           JOIN date_dim b ON b.date BETWEEN a.term_start AND a.term_end AND b.month_name = 'December' AND
	                              b.day_of_month = 31 AND b.year <= 2019
	           JOIN (SELECT isin, min(term_start) AS first_term FROM legislators_terms GROUP BY 1
	                )        c ON a.isin = c.isin
      GROUP BY 1, 2
     ) a
ORDER BY 1, 2
;

SELECT date
	 , coalesce(sum(CASE WHEN century = 18 THEN legislators END) * 100.0 / sum(legislators), 0) AS pct_18
	 , coalesce(sum(CASE WHEN century = 19 THEN legislators END) * 100.0 / sum(legislators), 0) AS pct_19
	 , coalesce(sum(CASE WHEN century = 20 THEN legislators END) * 100.0 / sum(legislators), 0) AS pct_20
	 , coalesce(sum(CASE WHEN century = 21 THEN legislators END) * 100.0 / sum(legislators), 0) AS pct_21
FROM (SELECT b.date, date_part('century', first_term)::int AS century, count(DISTINCT a.isin) AS legislators
      FROM legislators_terms a
	           JOIN date_dim b ON b.date BETWEEN a.term_start AND a.term_end AND b.month_name = 'December' AND
	                              b.day_of_month = 31 AND b.year <= 2019
	           JOIN (SELECT isin, min(term_start) AS first_term FROM legislators_terms GROUP BY 1
	                )        c ON a.isin = c.isin
      GROUP BY 1, 2
     ) aa
GROUP BY 1
ORDER BY 1
;

SELECT isin
	 , date
	 , count(date) OVER (PARTITION BY isin ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cume_years
FROM (SELECT DISTINCT a.isin, b.date
      FROM legislators_terms a
	           JOIN date_dim b ON b.date BETWEEN a.term_start AND a.term_end AND b.month_name = 'December' AND
	                              b.day_of_month = 31 AND b.year <= 2019
     ) a
;

SELECT date, cume_years, count(DISTINCT isin) AS legislators
FROM (SELECT isin
           , date
           , count(date)
             OVER (PARTITION BY isin ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cume_years
      FROM (SELECT DISTINCT a.isin, b.date
            FROM legislators_terms a
	                 JOIN date_dim b ON b.date BETWEEN a.term_start AND a.term_end AND b.month_name = 'December' AND
	                                    b.day_of_month = 31 AND b.year <= 2019
            GROUP BY 1, 2
           ) aa
     ) aaa
GROUP BY 1, 2
;

SELECT date, count(*) AS tenures
FROM (SELECT date, cume_years, count(DISTINCT isin) AS legislators
      FROM (SELECT isin
                 , date
                 , count(date)
                   OVER (PARTITION BY isin ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cume_years
            FROM (SELECT DISTINCT a.isin, b.date
                  FROM legislators_terms a
	                       JOIN date_dim b
	                            ON b.date BETWEEN a.term_start AND a.term_end AND b.month_name = 'December' AND
	                               b.day_of_month = 31 AND b.year <= 2019
                  GROUP BY 1, 2
                 ) aa
           ) aaa
      GROUP BY 1, 2
     ) aaaa
GROUP BY 1
;

SELECT date, tenure, legislators * 100.0 / sum(legislators) OVER (PARTITION BY date) AS pct_legislators
FROM (SELECT date
           , CASE
	             WHEN cume_years <= 4 THEN '1 to 4'
	             WHEN cume_years <= 10 THEN '5 to 10'
	             WHEN cume_years <= 20 THEN '11 to 20'
	             ELSE '21+' END   AS tenure
           , count(DISTINCT isin) AS legislators
      FROM (SELECT isin
                 , date
                 , count(date)
                   OVER (PARTITION BY isin ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cume_years
            FROM (SELECT DISTINCT a.isin, b.date
                  FROM legislators_terms a
	                       JOIN date_dim b
	                            ON b.date BETWEEN a.term_start AND a.term_end AND b.month_name = 'December' AND
	                               b.day_of_month = 31 AND b.year <= 2019
                  GROUP BY 1, 2
                 ) a
           ) aa
      GROUP BY 1, 2
     ) aaa
;