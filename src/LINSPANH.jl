# LINSPANH: Linear Spanish hydro-electric network, 97 variables, 33 equality constraints
# classification LNR2-MN-97-33
# Objective: minimize -X1 (linear)
# Constraints: 33 node flow balance equations (linear)
# Variable X1 also appears in objective with coeff -1.0
# All variables have default ubound=3024, lbound=0 unless specified
# Several variables are fixed (FX) or have custom bounds
#
# Node constraint matrix: each arc contributes +1 to head node, -1 to tail node
# Arcs: X1→N32→N1 (X1 goes from N32 with coeff -1, to N1 with coeff +1)
function linspanh_model(; T = Float64, backend = nothing, kwargs...)
    # Variable bounds (from BOUNDS section)
    # Default: lvar=0, uvar=3024
    lvar = fill(T(0), 97)
    uvar = fill(T(3024), 97)
    start = zeros(T, 97)

    # Fixed and special bounds from SIF BOUNDS section
    # Format: (index, lb, ub, start_val)
    bounds = [
        (1, T(77.0), T(77.01), T(77.0)),
        (2, T(1124.52), T(1124.53), T(1124.52)),
        (3, T(158.0), T(158.01), T(158.0)),
        (4, T(16.0), T(16.01), T(16.0)),
        (5, T(0), T(0), T(0)),        # FX
        (6, T(783.65), T(783.66), T(783.65)),
        (7, T(11.0), T(11.01), T(11.0)),
        (8, T(49.0), T(49.01), T(49.0)),
        (9, T(2155.17), T(2155.18), T(2155.17)),
        (10, T(252.0), T(252.01), T(252.0)),
        (11, T(0), T(397.84), T(51.38)),
        (12, T(0), T(222.32), T(140.21)),
        (13, T(0), T(205.63), T(142.79)),
        (14, T(0), T(205.63), T(21.91)),
        (15, T(0), T(205.63), T(164.7)),
        (16, T(0), T(124.83), T(58.19)),
        (17, T(0), T(127.01), T(58.19)),
        (18, T(0), T(61.08), T(61.08)),
        (19, T(0), T(614.84), T(566.43)),
        (20, T(0), T(778.08), T(583.57)),
        # X21..X24 default
        (25, T(0), T(7257.6), T(0)),
        (26, T(0), T(1209.6), T(0)),
        (27, T(0), T(907.2), T(0)),
        (28, T(0), T(7257.6), T(0)),
        (29, T(0), T(7257.6), T(0)),
        (30, T(77.0), T(77.0), T(77.0)),   # FX
        (31, T(403.4), T(1312.0), T(1049.53)),
        (32, T(158.0), T(158.0), T(158.0)), # FX
        (33, T(16.0), T(16.0), T(16.0)),    # FX
        (34, T(0), T(0), T(0)),             # FX
        (35, T(502.0), T(928.46), T(738.43)),
        (36, T(11.0), T(11.0), T(11.0)),    # FX
        (37, T(49.0), T(49.0), T(49.0)),    # FX
        (38, T(915.3), T(2611.6), T(1835.36)),
        (39, T(252.0), T(252.0), T(252.0)), # FX
        (40, T(0), T(397.84), T(32.06)),
        # X41..X44 default, start values:
        (41, T(0), T(222.32), T(0)),
        (42, T(0), T(205.63), T(4.2)),
        (43, T(0), T(205.63), T(48.37)),
        (44, T(0), T(205.63), T(52.57)),
        (45, T(0), T(124.83), T(59.85)),
        (46, T(0), T(127.01), T(59.85)),
        (47, T(0), T(61.08), T(58.24)),
        (48, T(0), T(614.84), T(0)),
        (49, T(0), T(778.08), T(18.76)),
        # X50..X53 default
        (54, T(0), T(7257.6), T(0)),
        (55, T(0), T(1209.6), T(0)),
        (56, T(0), T(907.2), T(0)),
        (57, T(0), T(7257.6), T(0)),
        (58, T(0), T(7257.6), T(0)),
        (59, T(77.0), T(77.0), T(77.0)),    # FX
        (60, T(403.4), T(1312.0), T(1081.87)),
        (61, T(158.0), T(158.0), T(158.0)), # FX
        (62, T(16.0), T(16.0), T(16.0)),    # FX
        (63, T(0), T(0), T(0)),             # FX
        (64, T(505.64), T(928.46), T(696.71)),
        (65, T(11.0), T(11.0), T(11.0)),    # FX
        (66, T(49.0), T(49.0), T(49.0)),    # FX
        (67, T(915.3), T(2611.6), T(1972.77)),
        (68, T(252.0), T(252.0), T(252.0)), # FX
        (69, T(0), T(397.84), T(18.13)),
        # X70..X72 default
        (72, T(0), T(205.63), T(18.13)),
        (73, T(0), T(205.63), T(18.13)),
        (74, T(0), T(124.83), T(5.81)),
        (75, T(0), T(127.01), T(5.81)),
        # X76, X77 default
        (78, T(0), T(778.08), T(6.02)),
        # X79..X82 default
        (83, T(0), T(7257.6), T(0)),
        (84, T(0), T(1209.6), T(0)),
        (85, T(0), T(907.2), T(0)),
        (86, T(0), T(7257.6), T(0)),
        (87, T(0), T(7257.6), T(0)),
        (88, T(77.0), T(77.01), T(77.0)),
        (89, T(1100.0), T(1100.01), T(1100.0)),
        (90, T(158.0), T(158.01), T(158.0)),
        (91, T(16.0), T(16.01), T(16.0)),
        (92, T(0), T(0), T(0)),             # FX
        (93, T(700.0), T(700.01), T(700.0)),
        (94, T(11.0), T(11.01), T(11.0)),
        (95, T(49.0), T(49.01), T(49.0)),
        (96, T(2000.0), T(2000.01), T(2000.0)),
        (97, T(252.0), T(252.01), T(252.0)),
    ]

    for (idx, lb, ub, sv) in bounds
        lvar[idx] = lb
        uvar[idx] = ub
        start[idx] = sv
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 97; lvar = lvar, uvar = uvar, start = start)

    # Objective: minimize -X1
    ExaModels.@add_obj(c, -x[1])

    # Node flow constraints (equality): sum of arc flows = demand at node
    # Each arc contributes with coefficient in two nodes
    # Node flow equation: sum_{arcs leaving node} x[k] - sum_{arcs entering node} x[k] + demand = 0
    # From CONSTANTS section: these are the RHS values
    # All constraints are equality (XE N(I))

    # Node constants (from SIF CONSTANTS section):
    # N1=-51.38, N2=-13.84, N3=-2.58, N4=-21.91, N6=-12.97, N8=-2.89
    # N9=-20.84, N10=-17.14, N11=-32.06, N12=-0.28, N13=-4.20, N14=-48.37
    # N16=-18.13, N18=1.61, N19=-26.6, N20=-18.76, N21=-18.13, N24=-18.13
    # N26=-9.10, N28=5.81, N29=-9.10, N30=-6.02, N31=608.35, N32=-4626.34, N33=4363.0
    # Nodes with no constant: 0.0 (N5, N7, N15, N17, N22, N23, N25, N27)

    # Arc-node incidence matrix: for each arc X(k), node_from gets -1, node_to gets +1
    # Arcs from VARIABLES section:
    # X1: N32→N1; X2: N32→N2; X3: N32→N3; X4: N32→N4; X5: N32→N5
    # X6: N32→N6; X7: N32→N7; X8: N32→N8; X9: N32→N9; X10: N32→N10
    # X11: N1→N2; X12: N2→N3; X13: N3→N5; X14: N4→N5; X15: N5→N9
    # X16: N6→N7; X17: N7→N8; X18: N8→N9; X19: N9→N10; X20: N10→N31
    # X21: N1→N2; X22: N2→N3; X23: N3→N9; X24: N4→N9; X25: N6→N7
    # X26: N7→N8; X27: N8→N9; X28: N9→N10; X29: N10→N31
    # X30: N1→N11; X31: N2→N12; X32: N3→N13; X33: N4→N14; X34: N5→N15
    # X35: N6→N16; X36: N7→N17; X37: N8→N18; X38: N9→N19; X39: N10→N20
    # X40: N11→N12; X41: N12→N13; X42: N13→N15; X43: N14→N15; X44: N15→N19
    # X45: N16→N17; X46: N17→N18; X47: N18→N19; X48: N19→N20; X49: N20→N31
    # X50: N11→N12; X51: N12→N13; X52: N13→N19; X53: N14→N19; X54: N16→N17
    # X55: N17→N18; X56: N18→N19; X57: N19→N20; X58: N20→N31
    # X59: N11→N21; X60: N12→N22; X61: N13→N23; X62: N14→N24; X63: N15→N25
    # X64: N16→N26; X65: N17→N27; X66: N18→N28; X67: N19→N29; X68: N20→N30
    # X69: N21→N22; X70: N22→N23; X71: N23→N25; X72: N24→N25; X73: N25→N29
    # X74: N26→N27; X75: N27→N28; X76: N28→N29; X77: N29→N30; X78: N30→N31
    # X79: N21→N22; X80: N22→N23; X81: N23→N29; X82: N24→N29; X83: N26→N27
    # X84: N27→N28; X85: N28→N29; X86: N29→N30; X87: N30→N31
    # X88: N21→N33; X89: N22→N33; X90: N23→N33; X91: N24→N33; X92: N25→N33
    # X93: N26→N33; X94: N27→N33; X95: N28→N33; X96: N29→N33; X97: N30→N33

    # Node demand RHS (from CONSTANTS, indexed 1..33)
    # Node numbering: N1→idx1, N2→2, ..., N10→10, N11→11, ..., N33→33
    rhs = zeros(T, 33)
    rhs[1] = T(-51.38); rhs[2] = T(-13.84); rhs[3] = T(-2.58); rhs[4] = T(-21.91)
    rhs[6] = T(-12.97); rhs[8] = T(-2.89); rhs[9] = T(-20.84); rhs[10] = T(-17.14)
    rhs[11] = T(-32.06); rhs[12] = T(-0.28); rhs[13] = T(-4.20); rhs[14] = T(-48.37)
    rhs[16] = T(-18.13); rhs[18] = T(1.61); rhs[19] = T(-26.6); rhs[20] = T(-18.76)
    rhs[21] = T(-18.13); rhs[24] = T(-18.13); rhs[26] = T(-9.10); rhs[28] = T(5.81)
    rhs[29] = T(-9.10); rhs[30] = T(-6.02); rhs[31] = T(608.35); rhs[32] = T(-4626.34)
    rhs[33] = T(4363.0)

    # Arc incidence: (arc_index, from_node, to_node)
    arcs = [
        (1,32,1), (2,32,2), (3,32,3), (4,32,4), (5,32,5),
        (6,32,6), (7,32,7), (8,32,8), (9,32,9), (10,32,10),
        (11,1,2), (12,2,3), (13,3,5), (14,4,5), (15,5,9),
        (16,6,7), (17,7,8), (18,8,9), (19,9,10), (20,10,31),
        (21,1,2), (22,2,3), (23,3,9), (24,4,9), (25,6,7),
        (26,7,8), (27,8,9), (28,9,10), (29,10,31),
        (30,1,11), (31,2,12), (32,3,13), (33,4,14), (34,5,15),
        (35,6,16), (36,7,17), (37,8,18), (38,9,19), (39,10,20),
        (40,11,12), (41,12,13), (42,13,15), (43,14,15), (44,15,19),
        (45,16,17), (46,17,18), (47,18,19), (48,19,20), (49,20,31),
        (50,11,12), (51,12,13), (52,13,19), (53,14,19), (54,16,17),
        (55,17,18), (56,18,19), (57,19,20), (58,20,31),
        (59,11,21), (60,12,22), (61,13,23), (62,14,24), (63,15,25),
        (64,16,26), (65,17,27), (66,18,28), (67,19,29), (68,20,30),
        (69,21,22), (70,22,23), (71,23,25), (72,24,25), (73,25,29),
        (74,26,27), (75,27,28), (76,28,29), (77,29,30), (78,30,31),
        (79,21,22), (80,22,23), (81,23,29), (82,24,29), (83,26,27),
        (84,27,28), (85,28,29), (86,29,30), (87,30,31),
        (88,21,33), (89,22,33), (90,23,33), (91,24,33), (92,25,33),
        (93,26,33), (94,27,33), (95,28,33), (96,29,33), (97,30,33),
    ]

    # Build node-arc matrix for constraint equations
    # Constraint for node n: sum_{arcs to n} x[k] - sum_{arcs from n} x[k] = rhs[n]
    # (inflow - outflow = demand, where demand is negative for sinks)
    # Actually in typical network: outflow - inflow = external injection (positive = source)
    # The SIF convention: each arc x[k] contributes X(from) -= 1.0, X(to) += 1.0 in constraint
    # So: sum_{arcs entering n} x[k] - sum_{arcs leaving n} x[k] = rhs[n]

    # Build lists per node
    entering = [Int[] for _ in 1:33]
    leaving = [Int[] for _ in 1:33]
    for (k, fn, tn) in arcs
        push!(leaving[fn], k)
        push!(entering[tn], k)
    end

    # Add constraints for each node
    for n in 1:33
        e = entering[n]
        l = leaving[n]
        if isempty(e) && isempty(l)
            continue
        end
        ExaModels.@add_con(c,
            sum(x[k] for k in e; init = T(0)) - sum(x[k] for k in l; init = T(0)) - rhs[n])
    end

    return ExaModels.ExaModel(c; kwargs...)
end
