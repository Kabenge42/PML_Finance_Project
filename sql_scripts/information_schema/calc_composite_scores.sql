create function information_schema.calc_composite_scores(p_isin text DEFAULT NULL::text)
    returns TABLE
            (
                isin                   text,
                piotroski_f_score      integer,
                dilution_score         numeric,
                quality_momentum_score numeric
            )
    stable
    parallel safe
    language plpgsql
as
$$
BEGIN
    RETURN QUERY
        SELECT p.isin,
               p.piotroski_f_score,
               d.dilution_score,
               q.quality_momentum_score
        FROM public.calc_piotroski_f_score(p_isin) p
                 JOIN public.calc_shareholder_dilution_features(p_isin) d ON p.isin = d.isin
                 JOIN public.calc_quality_momentum_composite(p_isin) q ON p.isin = q.isin;
END;
$$;

alter function information_schema.calc_composite_scores(unknown) owner to postgres;

