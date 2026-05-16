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
include("DIAGPQE.jl")
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
include("SREADIN3.jl")
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
include("CLPLATEC.jl")
include("CORKSCRW.jl")
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
include("CVXBQP1.jl")
include("CVXQP1.jl")
include("CVXQP2.jl")
include("COOLHANS.jl")
include("CAMSHAPE.jl")
include("COSHFUN.jl")
include("CATMIX.jl")
include("PDE1.jl")
include("PDE2.jl")
include("PRIMAL1.jl")
include("PRIMAL2.jl")
include("PRIMAL3.jl")
include("PRIMAL4.jl")
include("PRIMALC1.jl")
include("PRIMALC2.jl")
include("PRIMALC5.jl")
include("PRIMALC8.jl")
include("PRODPL0.jl")
include("PRODPL1.jl")
include("PT.jl")
include("DEGENQP.jl")
include("BQP1VAR.jl")
include("HYPCIR.jl")
include("HEART6.jl")
include("HEART8.jl")
include("OBSTCLAL.jl")
include("OBSTCLBL.jl")
include("OBSTCLBM.jl")
include("OBSTCLBU.jl")
include("ODFITS.jl")
include("OET2.jl")
include("OET4.jl")
include("OET5.jl")
include("OET6.jl")
include("OET7.jl")
include("OPTCDEG2.jl")
include("OPTCDEG3.jl")
include("OPTCTRL3.jl")
include("OPTCTRL6.jl")
include("OPTMASS.jl")
include("OPTPRLOC.jl")
include("OSCIPANE.jl")
include("OSLBQP.jl")
include("ORTHRDM2.jl")
include("ORTHRDS2.jl")
include("ORTHREGA.jl")
include("ORTHREGB.jl")
include("ORTHREGC.jl")
include("ORTHREGD.jl")
include("ORTHREGE.jl")
include("ORTHREGF.jl")
include("ORTHRGDM.jl")
include("ORTHRGDS.jl")
include("PENALTY3.jl")
include("PENTDI.jl")
include("POLYGON.jl")
include("POROUS1.jl")
include("POROUS2.jl")
include("PORTFL1.jl")
include("PORTFL2.jl")
include("PORTFL3.jl")
include("PORTFL4.jl")
include("PORTFL6.jl")
include("PORTSNQP.jl")
include("PORTSQP.jl")
include("POWELL20.jl")
include("POWELLBC.jl")
include("PROBPENL.jl")
include("QPBAND.jl")
include("QPNBAND.jl")
include("QR3D.jl")
include("QR3DBD.jl")
include("QR3DLS.jl")
include("QUDLIN.jl")
include("RAYBENDL.jl")
include("BLOWEYA.jl")
include("BLOWEYB.jl")
include("BLOWEYC.jl")
include("BQPGABIM.jl")
include("BQPGASIM.jl")
include("CHARDIS0.jl")
include("CHARDIS1.jl")
include("CHENHARK.jl")
include("CONT5-QP.jl")
include("CONT6-QQ.jl")
include("DECONVB.jl")
include("DECONVC.jl")
include("DECONVNE.jl")
include("DECONVU.jl")
include("DEGENQPC.jl")
include("DEMBO7.jl")
include("DNIEPER.jl")
include("AIRPORT.jl")
include("ALJAZZAF.jl")
include("ALLINQP.jl")
include("ANTWERP.jl")
include("ARGLALE.jl")
include("ARGLBLE.jl")
include("ARGLCLE.jl")
include("BLOCKQP1.jl")
include("BLOCKQP2.jl")
include("BRATU3D.jl")
include("BURKEHAN.jl")
include("CBRATU2D.jl")
include("CBRATU3D.jl")
include("CHEMRCTA.jl")
include("CHEMRCTB.jl")
include("CONGIGMZ.jl")
include("CVXQP3.jl")
include("CYCLIC3.jl")
include("DEGENLPA.jl")
include("DEGENLPB.jl")
include("DIPIGRI.jl")
include("DRCAV1LQ.jl")
include("DRCAV2LQ.jl")
include("DRCAV3LQ.jl")
include("DRCAVTY1.jl")
include("DRCAVTY2.jl")
include("DRCAVTY3.jl")
include("DTOC1L.jl")
include("DTOC1NA.jl")
include("DTOC1ND.jl")
include("DTOC2.jl")
include("DTOC3.jl")
include("DTOC4.jl")
include("DTOC5.jl")
include("DTOC6.jl")
include("DUAL1.jl")
include("DUAL2.jl")
include("DUAL3.jl")
include("DUAL4.jl")
include("EG3.jl")
include("EIGENA.jl")
include("EIGENA2.jl")
include("EIGENACO.jl")
include("EIGENALS.jl")
include("EIGENAU.jl")
include("EIGENB.jl")
include("EIGENB2.jl")
include("EIGENBCO.jl")
include("EIGENBLS.jl")
include("EIGENC.jl")
include("EIGENC2.jl")
include("EIGENCCO.jl")
include("EIGENCLS.jl")
include("EIGMAXA.jl")
include("EIGMAXB.jl")
include("EIGMAXC.jl")
include("EIGMINA.jl")
include("EIGMINB.jl")
include("EIGMINC.jl")
include("ELATTAR.jl")
include("ELEC.jl")
include("ERRINBAR.jl")
include("ERRINRSM.jl")
include("EXPFITA.jl")
include("EXPFITB.jl")
include("EXPFITC.jl")
include("FCCU.jl")
include("FLETBV3M.jl")
include("FLETCHER.jl")
include("FLT.jl")
include("FMINSRF2.jl")
include("FMINSURF.jl")
include("FLOSP2HH.jl")
include("GASOIL.jl")
include("GAUSSELM.jl")
include("GLIDER.jl")
include("GRIDNETA.jl")
include("GOULDQP1.jl")
include("GOULDQP2.jl")
include("GOULDQP3.jl")
include("GPP.jl")
include("GROUPING.jl")
include("HADAMALS.jl")
include("HADAMARD.jl")
include("HAGER4.jl")
include("HAIFAS.jl")
include("HAIFAM.jl")
include("HAIFAL.jl")
include("HALDMADS.jl")
include("HANGING.jl")
include("HARKERP2.jl")
include("HET_Z.jl")
include("HIE1327D.jl")
include("HIE1372D.jl")
include("HIER13.jl")
include("HIER133A.jl")
include("HIER133B.jl")
include("HIER133C.jl")
include("HIER133D.jl")
include("HIER133E.jl")
include("HIER16.jl")
include("HIER163A.jl")
include("HIER163B.jl")
include("HIER163C.jl")
include("HIER163D.jl")
include("HIER163E.jl")
include("HOLMES.jl")
include("HS99EXP.jl")
include("HUES_MOD.jl")
include("HUESTIS.jl")
include("HVYCRASH.jl")
include("HYDROELL.jl")
include("HYDROELM.jl")
include("HYDROELS.jl")
include("INTEGREQ.jl")
include("JANNSON3.jl")
include("JANNSON4.jl")
include("JNLBRNG1.jl")
include("JNLBRNG2.jl")
include("JNLBRNGA.jl")
include("JNLBRNGB.jl")
include("JUNKTURN.jl")
include("KISSING.jl")
include("KISSING2.jl")
include("KIWCRESC.jl")
include("KOEBHELB.jl")
include("KOWOSB.jl")
include("KSIP.jl")
include("KSS.jl")
include("LAKES.jl")
include("LEWISPOL.jl")
include("LCH.jl")
include("LHAIFAM.jl")
include("LINSPANH.jl")
include("LINVERSE.jl")
include("LIPPERT1.jl")
include("LIPPERT2.jl")
include("LMINSURF.jl")
include("LOADBAL.jl")
include("LOGHAIRY.jl")
include("LOOTSMA.jl")
include("LSNNODOC.jl")
include("LUKVLE14.jl")
include("LUKVLI14.jl")
include("MADSSCHJ.jl")
include("MANNE.jl")
include("MATRIX2.jl")
include("MCONCON.jl")
include("MINSURF.jl")
include("MINSURFO.jl")
include("MSQRTA.jl")
include("MSQRTB.jl")
include("MSQRTBLS.jl")
include("MWRIGHT.jl")
include("NASH.jl")
include("NCVXBQP.jl")
include("NGONE.jl")
include("NLMSURF.jl")
include("NOBNDTOR.jl")
include("NONMSQRT.jl")
include("OBSTCLAE.jl")
include("OSCIGRAD.jl")
include("READING1.jl")
include("READING2.jl")
include("READING3.jl")
include("READING4.jl")
include("READING5.jl")
include("READING678.jl")
include("READING9.jl")
include("RES.jl")
include("RK23.jl")
include("ROBOT.jl")
include("ROBOTARM.jl")
include("ROCKET.jl")
include("ROTDISC.jl")
include("S268.jl")
include("S277_280.jl")
include("S308.jl")
include("S316_322.jl")
include("S365.jl")
include("S365MOD.jl")
include("SCOND1LS.jl")
include("SEMICN2U.jl")
include("SEMICON1.jl")
include("SEMICON2.jl")
include("SENSORS.jl")
include("SIM2BQP.jl")
include("SIMBQP.jl")
include("SIMPLLPA.jl")
include("SIMPLLPB.jl")
include("SIPOW1M.jl")
include("SIPOW2M.jl")
include("SNAIL.jl")
include("SOSQP1.jl")
include("SOSQP2.jl")
include("SPIN.jl")
include("SPIN2.jl")
include("SPIN2OP.jl")
include("SPINOP.jl")
include("SSC.jl")
include("SSCOSINE.jl")
include("SSNLBEAM.jl")
include("STEENBRA.jl")
include("STEENBRB.jl")
include("STEERING.jl")
include("SUPERSIM.jl")
include("TENBARS1.jl")
include("TFI1.jl")
include("TFI2.jl")
include("TFI3.jl")
include("TOINTQOR.jl")
include("TORSION1.jl")
include("TORSIONA.jl")
include("TRAINF.jl")
include("TRAINH.jl")
include("TRIGGER.jl")
include("TRUSPYR1.jl")
include("TRY-B.jl")
include("TWOD.jl")
include("UBH1.jl")
include("VANDERM1.jl")
include("VIBRBEAM.jl")
include("WACHBIEG.jl")
include("WATER.jl")
include("WEEDS.jl")
include("WOMFLET.jl")
include("WOODSNE.jl")
include("YAO.jl")
include("YFIT.jl")
include("ZIGZAG.jl")
include("ZY2.jl")
include("BATCH.jl")
include("SYNTHES1.jl")
include("SYNTHES2.jl")
include("SYNTHES3.jl")
include("TRIMLOSS.jl")
include("DUALC1.jl")
include("DUALC2.jl")
include("DUALC5.jl")
include("DUALC8.jl")
include("AGG.jl")
include("QPCBLEND.jl")
include("QPNBLEND.jl")
include("QPCSTAIR.jl")
include("QPNSTAIR.jl")
include("QPCBOEI1.jl")
include("QPNBOEI1.jl")
include("QPCBOEI2.jl")
include("QPNBOEI2.jl")
include("NYSTROM5.jl")
include("NUFFIELD.jl")
include("MOSARQP1.jl")
include("MOSARQP2.jl")
include("LUBRIF.jl")
include("LUBRIFC.jl")
include("AUG2D.jl")
include("AUG3D.jl")
include("MSS1.jl")
include("MSS2.jl")
include("MSS3.jl")
include("NCB20.jl")
include("NCB20B.jl")
include("LEUVEN1.jl")
include("AVION2.jl")
include("DISC2.jl")
include("DISCS.jl")
include("SVANBERG.jl")
include("SMBANK.jl")
include("SPECAN.jl")
include("SPMSQRT.jl")
include("SPMSRTLS.jl")
include("SSBRYBND.jl")
include("STCQP1.jl")
include("STCQP2.jl")
include("STNQP1.jl")
include("STNQP2.jl")
include("TABLE1.jl")
include("TABLE3.jl")
include("TABLE4.jl")
include("TABLE5.jl")
include("TABLE6.jl")
include("TABLE7.jl")
include("TABLE8.jl")
include("TARGUS.jl")
include("TOYSARAH.jl")
include("TWO5IN6.jl")
include("YATP2SQ.jl")
include("YORKNET.jl")
include("SAWPATH.jl")
include("RDW2D51F.jl")
include("MESH.jl")
include("MARINE.jl")
include("ZAMB2-8.jl")
include("ZAMB2-9.jl")
include("ZAMB2-10.jl")
include("ZAMB2-11.jl")
include("SSEBLIN.jl")
include("SSEBNLN.jl")
include("TRO3X3.jl")
include("TRO4X4.jl")
include("TRO5X5.jl")
include("TRO6X2.jl")
include("TRO11X3.jl")
include("TRO21X5.jl")
include("TRO41X9.jl")
include("LIN.jl")
include("DENSCHNBNE.jl")
include("DENSCHNCNE.jl")
include("DENSCHNDNE.jl")
include("DENSCHNENE.jl")
include("DENSCHNFNE.jl")
include("EGGCRATENE.jl")
include("EXPFITNE.jl")
include("HATFLDANE.jl")
include("HATFLDBNE.jl")
include("HATFLDCNE.jl")
include("HATFLDFLNE.jl")
include("HS1NE.jl")
include("HS2NE.jl")
include("SSINE.jl")
include("WAYSEA1NE.jl")
include("WAYSEA2NE.jl")
include("EXTROSNBNE.jl")
include("BROWNBSNE.jl")
include("BEALENE.jl")
include("BOX3NE.jl")
include("BDQRTICNE.jl")
include("HIMMELBCLS.jl")
include("EGGCRATE.jl")
include("HATFLDFLS.jl")
include("LIARWHDNE.jl")
include("PRICE3NE.jl")
include("QINGNE.jl")
include("ROSENBRTU.jl")
include("ROSEPETAL2.jl")
include("S308NE.jl")
include("SISSER2.jl")
include("SSI.jl")
include("STREG.jl")
include("STREGNE.jl")
include("TQUARTICNE.jl")

export rosenbr_model,
       hs1_model, hs2_model, hs3_model, hs4_model, hs5_model,
       hs1ne_model, hs2ne_model,
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
       diagpqe_model, diagnqe_model, diagiqe_model, diagpqb_model, diagnqb_model, diagpqt_model, diagiqb_model, diagiqt_model, diagnqt_model,
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
       expfitne_model,
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
       hatfldane_model, hatfldbne_model, hatfldcne_model, hatfldflne_model,
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
       sreadin3_model,
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
       denschnbne_model, denschncne_model, denschndne_model, denschnene_model, denschnfne_model,
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
       allinita_model, allinitc_model,
       cvxbqp1_model, cvxqp1_model, cvxqp2_model,
       coolhans_model, camshape_model, coshfun_model, catmix_model,
       pde1_model, pde2_model,
       primal1_model, primal2_model, primal3_model, primal4_model,
       primalc1_model, primalc2_model, primalc5_model, primalc8_model,
       prodpl0_model, prodpl1_model,
       pt_model,
       degenqp_model, bqp1var_model, hypcir_model,
       heart6_model, heart8_model,
       obstclal_model, obstclbl_model, obstclbm_model, obstclbu_model,
       odfits_model,
       oet2_model, oet4_model, oet5_model, oet6_model, oet7_model,
       optcdeg2_model, optcdeg3_model,
       optctrl3_model, optctrl6_model,
       optmass_model,
       optprloc_model,
       oscipane_model,
       oslbqp_model,
       orthrdm2_model, orthrds2_model,
       orthrega_model, orthregb_model, orthregc_model, orthregd_model,
       orthrege_model, orthregf_model,
       orthrgdm_model, orthrgds_model,
       penalty3_model,
       pentdi_model,
       polygon_model,
       porous1_model, porous2_model,
       portfl1_model, portfl2_model, portfl3_model, portfl4_model, portfl6_model,
       portsnqp_model, portsqp_model,
       powell20_model, powellbc_model,
       probpenl_model,
       qpband_model, qpnband_model,
       qr3d_model, qr3dbd_model, qr3dls_model,
       qudlin_model,
       raybendl_model,
       bloweya_model,
       bloweyb_model,
       bloweyc_model,
       bqpgabim_model,
       bqpgasim_model,
       chardis0_model,
       chardis1_model,
       chenhark_model,
       cont5_qp_model,
       cont6_qq_model,
       deconvb_model,
       deconvc_model,
       deconvne_model,
       deconvu_model,
       degenqpc_model,
       dembo7_model,
       dnieper_model,
       airport_model,
       aljazzaf_model,
       allinqp_model,
       antwerp_model,
       arglale_model, arglble_model, arglcle_model,
       blockqp1_model,
       blockqp2_model, blockqp3_model, blockqp4_model, blockqp5_model,
       bratu3d_model,
       burkehan_model,
       cbratu2d_model, cbratu3d_model,
       chemrcta_model, chemrctb_model,
       congigmz_model,
       cvxqp3_model,
       cyclic3_model,
       degenlpa_model, degenlpb_model,
       dipigri_model,
       drcav1lq_model, drcav2lq_model, drcav3lq_model,
       drcavty1_model, drcavty2_model, drcavty3_model,
       dtoc1l_model,
       dtoc1na_model, dtoc1nb_model, dtoc1nc_model,
       dtoc1nd_model,
       dtoc2_model, dtoc3_model, dtoc4_model, dtoc5_model, dtoc6_model,
       dual1_model, dual2_model, dual3_model, dual4_model,
       eg3_model,
       eigena_model, eigena2_model, eigenaco_model, eigenals_model, eigenau_model,
       eigenb_model, eigenb2_model, eigenbco_model, eigenbls_model,
       eigenc_model, eigenc2_model, eigencco_model, eigencls_model,
       eigmaxa_model, eigmaxb_model, eigmaxc_model,
       eigmina_model, eigminb_model, eigminc_model,
       elattar_model,
       elec_model,
       errinbar_model, errinrsm_model,
       expfita_model, expfitb_model, expfitc_model,
       fccu_model,
       fletbv3m_model, fletcher_model, flt_model,
       fminsrf2_model, fminsurf_model,
       flosp2hh_model, flosp2hl_model, flosp2hm_model,
       flosp2th_model, flosp2tl_model, flosp2tm_model,
       gasoil_model,
       gausselm_model,
       glider_model,
       gridneta_model, gridnetb_model, gridnetc_model,
       gridnetd_model, gridnete_model, gridnetf_model,
       gouldqp1_model, gouldqp2_model, gouldqp3_model,
       gpp_model,
       grouping_model,
       hadamals_model, hadamard_model,
       hager4_model,
       haifas_model, haifam_model, haifal_model,
       haldmads_model, hanging_model, harkerp2_model,
       het_z_model,
       hie1327d_model, hie1372d_model,
       hier13_model,
       hier133a_model, hier133b_model, hier133c_model, hier133d_model, hier133e_model,
       hier16_model,
       hier163a_model, hier163b_model, hier163c_model, hier163d_model, hier163e_model,
       holmes_model,
       hs99exp_model,
       hues_mod_model, huestis_model,
       hvycrash_model,
       hydroell_model, hydroelm_model, hydroels_model,
       integreq_model,
       jannson3_model, jannson4_model,
       jnlbrng1_model, jnlbrng2_model, jnlbrnga_model, jnlbrngb_model,
       junkturn_model,
       kissing_model, kissing2_model,
       kiwcresc_model,
       koebhelb_model, kowosb_model,
       ksip_model, kss_model,
       lakes_model,
       lewispol_model,
       lch_model,
       lhaifam_model,
       linspanh_model, linverse_model,
       lippert1_model, lippert2_model,
       lminsurf_model, loadbal_model,
       loghairy_model, lootsma_model,
       lsnnodoc_model,
       lukvle14_model, lukvli14_model,
       madsschj_model,
       manne_model, matrix2_model, mconcon_model,
       minsurf_model, minsurfo_model,
       msqrta_model, msqrtb_model, msqrtbls_model,
       mwright_model,
       nash_model,
       ncvxbqp1_model, ncvxbqp2_model, ncvxbqp3_model,
       ngone_model, nlmsurf_model, nobndtor_model, nonmsqrt_model,
       obstclae_model,
       oscigrad_model,
       reading1_model, reading2_model, reading3_model, reading4_model,
       reading5_model,
       reading6_model, reading7_model, reading8_model,
       reading9_model,
       res_model,
       rk23_model,
       robot_model, robotarm_model, rocket_model, rotdisc_model,
       s268_model, s277_280_model, s308_model, s316_322_model,
       s365_model, s365mod_model,
       scond1ls_model,
       semicn2u_model, semicon1_model, semicon2_model,
       sensors_model,
       sim2bqp_model, simbqp_model,
       simpllpa_model, simpllpb_model,
       sipow1m_model, sipow2m_model,
       snail_model,
       sosqp1_model, sosqp2_model,
       spin_model, spin2_model, spin2op_model, spinop_model,
       ssc_model, sscosine_model, ssnlbeam_model,
       steenbra_model,
       steenbrb_model, steenbrc_model, steenbrd_model, steenbre_model,
       steenbrf_model, steenbrg_model,
       steering_model,
       supersim_model,
       tenbars1_model, tenbars2_model, tenbars3_model, tenbars4_model,
       tfi1_model, tfi2_model, tfi3_model,
       tointqor_model,
       torsion1_model, torsion2_model, torsion3_model,
       torsion4_model, torsion5_model, torsion6_model,
       torsiona_model, torsionb_model, torsionc_model,
       torsiond_model, torsione_model, torsionf_model,
       trainf_model, trainh_model,
       trigger_model,
       truspyr1_model, truspyr2_model,
       try_b_model,
       twod_model,
       ubh1_model, ubh5_model,
       vanderm1_model, vanderm2_model, vanderm3_model, vanderm4_model,
       vibrbeam_model, wachbieg_model,
       water_model,
       weeds_model, womflet_model,
       woodsne_model,
       yao_model,
       yfit_model, yfitu_model, yfitne_model,
       zigzag_model,
       zy2_model,
       batch_model,
       synthes1_model, synthes2_model, synthes3_model,
       trimloss_model,
       dualc1_model, dualc2_model, dualc5_model, dualc8_model,
       agg_model,
       qpcblend_model, qpnblend_model,
       qpcstair_model, qpnstair_model,
       qpcboei1_model, qpnboei1_model,
       qpcboei2_model, qpnboei2_model,
       nystrom5_model,
       nuffield_model,
       mosarqp1_model, mosarqp2_model,
       lubrif_model, lubrifc_model,
       aug2d_model, aug2dc_model, aug2dqp_model, aug2dcqp_model,
       aug3d_model, aug3dc_model, aug3dqp_model, aug3dcqp_model,
       mss1_model, mss2_model, mss3_model,
       ncb20_model, ncb20b_model,
       leuven1_model, leuven2_model, leuven3_model, leuven4_model,
       leuven5_model, leuven6_model, leuven7_model,
       avion2_model,
       clplatec_model,
       corkscrw_model,
       disc2_model, discs_model,
       svanberg_model,
       smbank_model,
       yorknet_model,
       specan_model,
       spmsqrt_model,
       spmsrtls_model,
       ssbrybnd_model,
       stcqp1_model, stcqp2_model,
       stnqp1_model, stnqp2_model,
       table1_model, table3_model, table4_model, table5_model,
       table6_model, table7_model, table8_model,
       targus_model,
       toysarah_model,
       two5in6_model,
       yatp2sq_model,
       sawpath_model,
       rdw2d51f_model, rdw2d51u_model,
       rdw2d52b_model, rdw2d52f_model, rdw2d52u_model,
       mesh_model,
       marine_model,
       zamb2_8_model,
       zamb2_9_model,
       zamb2_10_model,
       zamb2_11_model,
       sseblin_model,
       ssebnln_model,
       tro3x3_model,
       tro4x4_model,
       tro5x5_model,
       tro6x2_model,
       tro11x3_model,
       tro21x5_model,
       tro41x9_model,
       lin_model,
       eggcratene_model, ssine_model, waysea1ne_model, waysea2ne_model,
       extrosnbne_model, brownbsne_model, bealene_model, box3ne_model, bdqrticne_model, himmelbcls_model,
       eggcrate_model, hatfldfls_model, liarwhdne_model, price3ne_model, qingne_model,
       rosenbrtu_model, rosepetal2_model, s308ne_model, sisser2_model, ssi_model,
       streg_model, stregne_model, tquarticne_model

end # module ExaCUTEst
