module ExaCUTEst

using ExaModels
using SpecialFunctions

include("ROSENBR.jl")
include("HS1.jl")
include("HS2.jl")
include("HS3.jl")
include("HS4.jl")
include("HS5.jl")
include("HS6.jl")
include("HS7.jl")
include("HS8.jl")
include("HS9.jl")
include("HS10.jl")
include("HS11.jl")
include("HS12.jl")
include("HS13.jl")
include("HS14.jl")
include("HS15.jl")
include("HS16.jl")
include("HS17.jl")
include("HS18.jl")
include("HS19.jl")
include("HS20.jl")
include("HS21.jl")
include("HS22.jl")
include("HS23.jl")
include("HS24.jl")
include("HS25.jl")
include("HS26.jl")
include("HS27.jl")
include("HS28.jl")
include("HS29.jl")
include("HS30.jl")
include("HS31.jl")
include("HS32.jl")
include("HS33.jl")
include("HS34.jl")
include("HS35.jl")
include("HS36.jl")
include("HS37.jl")
include("HS38.jl")
include("HS39.jl")
include("HS40.jl")
include("HS41.jl")
include("HS42.jl")
include("HS43.jl")
include("HS44.jl")
include("HS45.jl")
include("HS46.jl")
include("HS47.jl")
include("HS48.jl")
include("HS49.jl")
include("HS50.jl")
include("HS51.jl")
include("HS52.jl")
include("HS53.jl")
include("HS54.jl")
include("HS55.jl")
include("HS56.jl")
include("HS57.jl")
include("HS59.jl")
include("HS60.jl")
include("HS61.jl")
include("HS62.jl")
include("HS63.jl")
include("HS64.jl")
include("HS65.jl")
include("HS66.jl")
include("HS67.jl")
include("HS68.jl")
include("HS69.jl")
include("HS71.jl")
include("HS72.jl")
include("HS73.jl")
include("HS74.jl")
include("HS75.jl")
include("HS76.jl")
include("HS77.jl")
include("HS78.jl")
include("HS79.jl")
include("HS80.jl")
include("HS81.jl")
include("HS83.jl")
include("HS84.jl")
include("HS86.jl")
include("HS93.jl")
include("HS95.jl")
include("HS96.jl")
include("HS97.jl")
include("HS98.jl")
include("HS99.jl")
include("HS100.jl")
include("HS101.jl")
include("HS102.jl")
include("HS103.jl")
include("HS104.jl")
include("HS105.jl")
include("HS106.jl")
include("HS107.jl")
include("HS108.jl")
include("HS109.jl")
include("HS110.jl")
include("HS111.jl")
include("HS112.jl")
include("HS113.jl")
include("HS114.jl")
include("HS116.jl")
include("HS117.jl")
include("HS118.jl")
include("HS119.jl")
include("BYRDSPHR.jl")
include("TAME.jl")
include("ALSOTAME.jl")
include("DEGDIAG.jl")
include("ZANGWIL3.jl")
include("BOOTH.jl")
include("HIMMELBC.jl")
include("HIMMELBD.jl")
include("HIMMELBE.jl")
include("HIMMELBG.jl")
include("HIMMELBH.jl")
include("HIMMELP1.jl")
include("HIMMELP2.jl")
include("HIMMELP3.jl")
include("HIMMELP4.jl")
include("HIMMELP5.jl")
include("HIMMELP6.jl")
include("RSNBRNE.jl")
include("BT1.jl")
include("BT10.jl")
include("BT11.jl")
include("BT12.jl")
include("BT13.jl")
include("CB2.jl")
include("CB3.jl")
include("ZECEVIC2.jl")
include("ZECEVIC3.jl")
include("ZECEVIC4.jl")
include("DQDRTIC.jl")
include("POWER.jl")
include("EXPFIT.jl")
include("LSQFIT.jl")
include("PALMER1.jl")
include("PALMER2.jl")
include("PALMER3.jl")
include("PALMER4.jl")
include("PALMER1A.jl")
include("PALMER1B.jl")
include("PALMER1C.jl")
include("PALMER1D.jl")
include("PALMER1E.jl")
include("PALMER2A.jl")
include("PALMER2B.jl")
include("PALMER2C.jl")
include("PALMER2E.jl")
include("PALMER3A.jl")
include("PALMER3B.jl")
include("PALMER3C.jl")
include("PALMER3E.jl")
include("PALMER4A.jl")
include("PALMER4B.jl")
include("PALMER4C.jl")
include("PALMER4E.jl")
include("PALMER5A.jl")
include("PALMER5B.jl")
include("PALMER5C.jl")
include("PALMER5D.jl")
include("PALMER5E.jl")
include("PALMER6A.jl")
include("PALMER6C.jl")
include("PALMER6E.jl")
include("PALMER7A.jl")
include("PALMER7C.jl")
include("PALMER7E.jl")
include("PALMER8A.jl")
include("PALMER8C.jl")
include("PALMER8E.jl")
include("CHEBYQAD.jl")
include("CHNRSBNE.jl")
include("BEALE.jl")
include("SISSER.jl")
include("CUBE.jl")
include("POWELLSG.jl")
include("OSBORNEA.jl")
include("EDENSCH.jl")
include("HELIX.jl")
include("HIMMELBF.jl")
include("COSINE.jl")
include("TQUARTIC.jl")
include("TRIDIA.jl")
include("POWELLSQ.jl")
include("HS268.jl")
include("BARD.jl")
include("ENGVAL1.jl")
include("ARWHEAD.jl")
include("WOODS.jl")
include("BROWNBS.jl")
include("SINQUAD.jl")
include("DIXMAANA.jl")
include("DIXMAANB.jl")
include("DIXMAANC.jl")
include("DIXMAAND.jl")
include("DIXMAANE.jl")
include("DIXMAANF.jl")
include("DIXMAANG.jl")
include("DIXMAANH.jl")
include("DIXMAANI.jl")
include("DIXMAANJ.jl")
include("DIXMAANK.jl")
include("DIXMAANL.jl")
include("DIXMAANM.jl")
include("DIXMAANN.jl")
include("DIXMAANO.jl")
include("DIXMAANP.jl")
include("BIGGS3.jl")
include("BIGGS5.jl")
include("BIGGS6.jl")
include("BOX2.jl")
include("BOX3.jl")
include("HATFLDB.jl")
include("SCHMVETT.jl")
include("BDQRTIC.jl")
include("NONSCOMP.jl")
include("VARDIM.jl")
include("FREUROTH.jl")
include("NONCVXU2.jl")
include("SPARSQUR.jl")
include("BROYDN3D.jl")
include("BROYDN7D.jl")
include("BRYBND.jl")
include("ENGVAL2.jl")
include("CRAGGLVY.jl")
include("MOREBV.jl")
include("GENHUMPS.jl")
include("HIMMELBI.jl")
include("HIMMELBJ.jl")
include("CHNROSNB.jl")
include("FLETCBV2.jl")
include("HIMMELBK.jl")
include("BROWNDEN.jl")
include("JENSMP.jl")
include("MEXHAT.jl")
include("MODBEALE.jl")
include("OSBORNEB.jl")
include("SROSENBR.jl")
include("TESTQUAD.jl")
include("TOINTGSS.jl")
include("ZANGWIL2.jl")
include("MANCINO.jl")
include("MSQRTALS.jl")
include("PENALTY1.jl")
include("PENALTY2.jl")
include("FLETCHBV.jl")
include("QUARTC.jl")
include("EXPLIN.jl")
include("EXPLIN2.jl")
include("GENROSE.jl")
include("ARGAUSS.jl")
include("ARGLINA.jl")
include("HATFLDA.jl")
include("HATFLDC.jl")
include("HATFLDD.jl")
include("HATFLDE.jl")
include("HEART6LS.jl")
include("HEART8LS.jl")
include("HATFLDF.jl")
include("HATFLDFL.jl")
include("HATFLDH.jl")
include("SCURLY10.jl")
include("BDRY2.jl")
include("HILBERTA.jl")
include("BROWNALE.jl")
include("CLPLATEA.jl")
include("YATP1SQ.jl")
include("EXTROSNB.jl")
include("NONDIA.jl")
include("CHAINWOO.jl")
include("CLIFF.jl")
include("DENSCHNA.jl")
include("DENSCHNB.jl")
include("DENSCHNC.jl")
include("DENSCHND.jl")
include("DENSCHNE.jl")
include("DENSCHNF.jl")
include("NONDQUAR.jl")
include("INDEF.jl")
include("INDEFM.jl")
include("MADSEN.jl")
include("MAKELA1.jl")
include("MAKELA2.jl")
include("MAKELA3.jl")
include("MAKELA4.jl")
include("GOTTFR.jl")
include("SCOSINE.jl")
include("SINEALI.jl")
include("S368.jl")
include("GULF.jl")
include("CAMEL6.jl")
include("BRKMCC.jl")
include("ALLINITU.jl")
include("HS70.jl")
include("BT2.jl")
include("BT3.jl")
include("BT4.jl")
include("BT5.jl")
include("BT6.jl")
include("BT7.jl")
include("BT8.jl")
include("BT9.jl")
include("CHNRSNBM.jl")
include("ARWHDNE.jl")
include("PENTAGON.jl")
include("CANTILVR.jl")
include("HUMPS.jl")
include("BIGGSC4.jl")
include("GOFFIN.jl")
include("MIFFLIN1.jl")
include("MIFFLIN2.jl")
include("OSCIPATH.jl")
include("WATSON.jl")
include("BDEXP.jl")
include("ARTIF.jl")
include("FLETCBV3.jl")
include("EXPQUAD.jl")
include("QRTQUAD.jl")
include("OSCIGRNE.jl")
include("SINROSNB.jl")
include("NONCVXUN.jl")
include("CATENA.jl")
include("CATENARY.jl")
include("BIGGSB1.jl")
include("CURLY10.jl")
include("CURLY20.jl")
include("CURLY30.jl")
include("BROYDNBD.jl")
include("CLNLBEAM.jl")
include("CHAIN.jl")
include("CLPLATEB.jl")
include("DQRTIC.jl")
include("DEGTRID.jl")
include("DEGTRID2.jl")
include("DEGTRIDL.jl")
include("ARGTRIG.jl")
include("ARGLINB.jl")
include("ARGLINC.jl")
include("SCURLY20.jl")
include("SCURLY30.jl")
include("FLETCHCR.jl")
include("GENHS28.jl")
include("GILBERT.jl")
include("LUKVLE1.jl")
include("LUKVLE2.jl")
include("LUKVLE3.jl")
include("LUKVLE4.jl")
include("LUKVLE5.jl")
include("LUKVLE6.jl")
include("LUKVLE7.jl")
include("LUKVLE8.jl")
include("LUKVLE9.jl")
include("LUKVLE10.jl")
include("LUKVLE11.jl")
include("LUKVLE12.jl")
include("LUKVLE13.jl")
# LUKVLE14.jl excluded: CUTEst sifdecoder applies pervasive variable swaps (~2220 positions)
# that cannot be replicated. Nearly all callbacks fail. Re-enable when the swap is decoded.
# include("LUKVLE14.jl")
include("LUKVLE15.jl")
include("LUKVLE16.jl")
include("LUKVLE17.jl")
include("LUKVLE18.jl")
include("LUKVLI1.jl")
include("LUKVLI2.jl")
include("LUKVLI3.jl")
include("LUKVLI4.jl")
include("LUKVLI5.jl")
include("LUKVLI6.jl")
include("LUKVLI7.jl")
include("LUKVLI8.jl")
include("LUKVLI9.jl")
include("LUKVLI10.jl")
include("LUKVLI11.jl")
include("LUKVLI12.jl")
include("LUKVLI13.jl")
# LUKVLI14.jl excluded: same sifdecoder variable-swap issue as LUKVLE14.
# include("LUKVLI14.jl")
include("LUKVLI15.jl")
include("LUKVLI16.jl")
include("LUKVLI17.jl")
include("LUKVLI18.jl")
include("BRATU1D.jl")
include("BRATU2D.jl")
include("BRATU2DT.jl")
include("LIARWHD.jl")
include("SINEVAL.jl")
include("SPARSINE.jl")
include("BOXPOWER.jl")
include("HILBERTB.jl")
include("SBRYBND.jl")
include("DIXON3DQ.jl")
include("MARATOSB.jl")
include("HIMMELBA.jl")
include("HIMMELBB.jl")
include("BROWNAL.jl")
include("MEYER3.jl")
include("LOGROS.jl")
include("CSFI1.jl")
include("CSFI2.jl")
include("LISWET1.jl")
include("LISWET2.jl")
include("LISWET3.jl")
include("LISWET4.jl")
include("LISWET5.jl")
include("LISWET6.jl")
include("LISWET7.jl")
include("LISWET8.jl")
include("LISWET9.jl")
include("LISWET10.jl")
include("LISWET11.jl")
include("LISWET12.jl")
include("GROWTHLS.jl")
include("NCVXQP.jl")
include("BOX.jl")
include("CHACONN1.jl")
include("CHACONN2.jl")
include("AIRCRFTA.jl")
include("AIRCRFTB.jl")
include("BDVALUE.jl")
include("BDVALUES.jl")
include("CHANDHEQ.jl")
include("CHANDHEU.jl")
include("DIXCHLNG.jl")
include("DIXCHLNV.jl")
include("DEMYMALO.jl")
include("GIGOMEZ1.jl")
include("GIGOMEZ2.jl")
include("GIGOMEZ3.jl")
include("CLUSTER.jl")
include("CUBENE.jl")
include("EXTRASIM.jl")
include("ERRINROS.jl")
include("EG1.jl")
include("EG2.jl")
include("GROWTH.jl")
include("HAGER1.jl")
include("HAGER2.jl")
include("HAGER3.jl")
include("MARATOS.jl")
include("MINMAXRB.jl")
include("MCCORMCK.jl")
include("MINMAXBD.jl")
include("OET1.jl")
include("OET3.jl")
include("SNAKE.jl")
include("SIPOW1.jl")
include("SIPOW2.jl")
include("STANCMIN.jl")
include("SINVALNE.jl")
include("ROSENMMX.jl")
include("POLAK1.jl")
include("POLAK2.jl")
include("POLAK3.jl")
include("POLAK4.jl")
include("POLAK5.jl")
include("POLAK6.jl")
include("SIPOW3.jl")
include("SIPOW4.jl")
include("SPIRAL.jl")
include("TWOBARS.jl")
include("RECIPE.jl")
include("POWELLBS.jl")
include("PSPDOC.jl")
include("HART6.jl")
include("HAIRY.jl")
include("HS3MOD.jl")
include("HS21MOD.jl")
include("HS35I.jl")
include("HS35MOD.jl")
include("HS44NEW.jl")
include("HS76I.jl")
include("HONG.jl")
include("HATFLDG.jl")
include("MDHOLE.jl")
include("MISTAKE.jl")
include("LOTSCHD.jl")
include("ALLINIT.jl")
include("OPTCNTRL.jl")
include("AVGASA.jl")
include("AVGASB.jl")
include("PFIT1.jl")
include("PFIT1LS.jl")
include("PFIT2.jl")
include("PFIT2LS.jl")
include("PFIT3.jl")
include("PFIT3LS.jl")
include("PFIT4.jl")
include("PFIT4LS.jl")
include("HS100LNP.jl")
include("HS100MOD.jl")
include("HS111LNP.jl")
include("ALLINITA.jl")
include("ALLINITC.jl")

export rosenbr_model,
       hs1_model, hs2_model, hs3_model, hs4_model, hs5_model,
       hs6_model, hs7_model, hs8_model, hs9_model, hs10_model,
       hs11_model, hs12_model, hs13_model, hs14_model, hs15_model,
       hs16_model, hs17_model, hs18_model, hs19_model, hs20_model,
       hs21_model, hs22_model, hs23_model, hs24_model,
       hs25_model, hs26_model, hs27_model, hs28_model, hs29_model, hs30_model,
       hs31_model, hs32_model, hs33_model, hs34_model,
       hs35_model, hs36_model, hs37_model, hs38_model,
       hs39_model, hs40_model, hs41_model, hs42_model, hs43_model, hs44_model, hs45_model,
       hs46_model, hs47_model, hs48_model, hs49_model, hs50_model,
       hs51_model, hs52_model, hs53_model, hs54_model, hs55_model,
       hs56_model, hs57_model, hs59_model,
       hs60_model, hs61_model, hs62_model, hs63_model, hs64_model,
       hs65_model, hs66_model,
       hs67_model, hs68_model, hs69_model,
       hs71_model, hs72_model, hs73_model, hs74_model, hs75_model,
       hs76_model, hs77_model, hs78_model, hs79_model, hs80_model,
       hs81_model, hs83_model, hs84_model, hs86_model, hs93_model,
       hs95_model, hs96_model, hs97_model, hs98_model, hs99_model,
       hs100_model, hs101_model, hs102_model, hs103_model,
       hs104_model, hs105_model, hs106_model, hs107_model, hs108_model, hs109_model, hs110_model, hs111_model, hs112_model,
       hs113_model, hs114_model, hs116_model, hs117_model, hs118_model, hs119_model,
       byrdsphr_model,
       tame_model, alsotame_model, degdiag_model,
       zangwil3_model, booth_model, himmelbc_model, himmelbd_model, himmelbe_model,
       himmelbg_model, himmelbh_model,
       himmelp1_model, himmelp2_model, himmelp3_model,
       himmelp4_model, himmelp5_model, himmelp6_model,
       rsnbrne_model, bt1_model, bt2_model, bt3_model, bt4_model, bt5_model,
       bt6_model, bt7_model, bt8_model, bt9_model,
       bt10_model, bt11_model, bt12_model, bt13_model,
       cb2_model, cb3_model,
       zecevic2_model,
       zecevic3_model,
       zecevic4_model,
       dqdrtic_model,
       power_model,
       expfit_model, lsqfit_model,
       palmer1_model, palmer2_model, palmer3_model, palmer4_model,
       palmer1a_model, palmer1b_model, palmer1c_model, palmer1d_model, palmer1e_model,
       palmer2a_model, palmer2b_model, palmer2c_model, palmer2e_model,
       palmer3a_model, palmer3b_model, palmer3c_model, palmer3e_model,
       palmer4a_model, palmer4b_model, palmer4c_model, palmer4e_model,
       palmer5a_model, palmer5b_model, palmer5c_model, palmer5d_model, palmer5e_model,
       palmer6a_model, palmer6c_model, palmer6e_model,
       palmer7a_model, palmer7c_model, palmer7e_model,
       palmer8a_model, palmer8c_model, palmer8e_model,
       chebyqad_model,
       chnrsbne_model,
       beale_model, sisser_model, cube_model,
       powellsg_model,
       osbornea_model,
       edensch_model,
       helix_model,
       himmelbf_model,
       cosine_model,
       tquartic_model,
       tridia_model,
       powellsq_model,
       hs268_model,
       bard_model,
       engval1_model,
       arwhead_model,
       dixmaana_model,
       dixmaanb_model,
       dixmaanc_model,
       dixmaand_model,
       dixmaane_model,
       dixmaanf_model,
       dixmaang_model,
       dixmaanh_model,
       dixmaani_model,
       dixmaanj_model,
       dixmaank_model,
       dixmaanl_model,
       dixmaanm_model,
       dixmaann_model,
       dixmaano_model,
       dixmaanp_model,
       woods_model,
       brownbs_model,
       sinquad_model,
       biggs3_model, biggs5_model, biggs6_model,
       box2_model, box3_model,
       hatfldb_model,
       schmvett_model,
       bdqrtic_model,
       nonscomp_model,
       vardim_model,
       freuroth_model,
       noncvxu2_model,
       sparsqur_model,
       broydn3d_model,
       broydn7d_model,
       brybnd_model,
       engval2_model,
       cragglvy_model,
       morebv_model,
       genhumps_model,
       himmelbi_model,
       himmelbj_model,
       chnrosnb_model,
       fletcbv2_model,
       himmelbk_model,
       brownden_model,
       jensmp_model,
       mexhat_model,
       modbeale_model,
       osborneb_model,
       srosenbr_model,
       testquad_model,
       tointgss_model,
       zangwil2_model,
       mancino_model,
       msqrtals_model,
       penalty1_model, penalty2_model,
       fletchbv_model, quartc_model, explin_model, explin2_model, genrose_model,
       argauss_model, arglina_model, hatflda_model, hatfldc_model, hatfldd_model, hatflde_model, hatfldf_model, hatfldfl_model, hatfldh_model, scurly10_model,
       heart6ls_model, heart8ls_model,
       bdry2_model, hilberta_model,
       brownale_model,
       clplatea_model,
       yatp1sq_model,
       extrosnb_model,
       nondia_model,
       chainwoo_model,
       cliff_model,
       denschna_model,
       denschnb_model,
       denschnc_model,
       denschnd_model,
       denschne_model,
       denschnf_model,
       nondquar_model,
       indef_model,
       indefm_model,
       madsen_model,
       makela1_model,
       makela2_model,
       makela3_model,
       makela4_model,
       gottfr_model,
       scosine_model,
       sineali_model,
       s368_model,
       gulf_model,
       camel6_model,
       brkmcc_model,
       allinitu_model,
       hs70_model,
       chnrsnbm_model,
       arwhdne_model,
       pentagon_model,
       cantilvr_model,
       humps_model,
       biggsc4_model,
       goffin_model,
       mifflin1_model,
       mifflin2_model,
       oscipath_model,
       watson_model,
       bdexp_model,
       artif_model,
       fletcbv3_model,
       expquad_model,
       qrtquad_model,
       oscigrne_model,
       sinrosnb_model,
       noncvxun_model,
       catena_model,
       catenary_model,
       biggsb1_model,
       curly10_model,
       curly20_model,
       curly30_model,
       broydnbd_model,
       clnlbeam_model,
       chain_model,
       clplateb_model,
       dqrtic_model,
       degtrid_model,
       degtrid2_model,
       degtridl_model,
       argtrig_model,
       arglinb_model,
       arglinc_model,
       scurly20_model,
       scurly30_model,
       fletchcr_model,
       genhs28_model,
       gilbert_model,
       lukvle1_model,
       lukvle2_model,
       lukvle3_model,
       lukvle4_model,
       lukvle5_model,
       lukvle6_model,
       lukvle7_model,
       lukvle8_model,
       lukvle9_model,
       lukvle10_model,
       lukvle11_model,
       lukvle12_model,
       lukvle13_model,
       # lukvle14_model excluded (sifdecoder variable swaps)
       lukvle15_model,
       lukvle16_model,
       lukvle17_model,
       lukvle18_model,
       lukvli1_model,
       lukvli2_model,
       lukvli3_model,
       lukvli4_model,
       lukvli5_model,
       lukvli6_model,
       lukvli7_model,
       lukvli8_model,
       lukvli9_model,
       lukvli10_model,
       lukvli11_model,
       lukvli12_model,
       lukvli13_model,
       # lukvli14_model excluded (same sifdecoder variable-swap issue)
       lukvli15_model,
       lukvli16_model,
       lukvli17_model,
       lukvli18_model,
       bratu1d_model,
       bratu2d_model,
       bratu2dt_model,
       liarwhd_model,
       sineval_model,
       sparsine_model,
       boxpower_model,
       hilbertb_model,
       sbrybnd_model,
       dixon3dq_model,
       maratosb_model,
       himmelba_model,
       himmelbb_model,
       brownal_model,
       meyer3_model,
       logros_model,
       csfi1_model,
       csfi2_model,
       liswet1_model,
       liswet2_model,
       liswet3_model,
       liswet4_model,
       liswet5_model,
       liswet6_model,
       liswet7_model,
       liswet8_model,
       liswet9_model,
       liswet10_model,
       liswet11_model,
       liswet12_model,
       growthls_model,
       ncvxqp1_model, ncvxqp2_model, ncvxqp3_model,
       ncvxqp4_model, ncvxqp5_model, ncvxqp6_model,
       ncvxqp7_model, ncvxqp8_model, ncvxqp9_model,
       box_model,
       chaconn1_model, chaconn2_model,
       aircrfta_model, aircrftb_model,
       bdvalue_model, bdvalues_model,
       chandheq_model, chandheu_model,
       dixchlng_model, dixchlnv_model,
       demymalo_model,
       gigomez1_model, gigomez2_model, gigomez3_model,
       cluster_model, cubene_model, extrasim_model,
       errinros_model, eg1_model, eg2_model,
       growth_model,
       hager1_model, hager2_model, hager3_model,
       maratos_model, minmaxrb_model, mccormck_model, minmaxbd_model,
       oet1_model, oet3_model,
       snake_model, sipow1_model, sipow2_model,
       stancmin_model, sinvalne_model,
       rosenmmx_model,
       polak1_model, polak2_model, polak3_model,
       polak4_model, polak5_model, polak6_model,
       sipow3_model, sipow4_model,
       spiral_model, twobars_model, recipe_model,
       powellbs_model, pspdoc_model,
       hart6_model, hairy_model,
       hs3mod_model, hs21mod_model, hs35i_model, hs35mod_model,
       hs44new_model, hs76i_model, hong_model,
       hatfldg_model, mdhole_model,
       mistake_model,
       lotschd_model, allinit_model, optcntrl_model,
       avgasa_model, avgasb_model,
       pfit1_model, pfit1ls_model,
       pfit2_model, pfit2ls_model,
       pfit3_model, pfit3ls_model,
       pfit4_model, pfit4ls_model,
       hs100lnp_model, hs100mod_model, hs111lnp_model,
       allinita_model, allinitc_model

end # module ExaCUTEst
