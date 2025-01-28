from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from database.db_history_loader import insert_db
from database.mysql_connector import get_db_connection
from Main import trials_extraction
import json
import pandas as pd

# Initialize the FastAPI application
app = FastAPI()

# CORS middleware configuration to allow requests from any origin
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware for DB connection handling (optional)
@app.middleware("http")
async def db_connection_middleware(request: Request, call_next):
    response = await call_next(request)
    return response

# Endpoint to fetch distinct NCT numbers
@app.get("/api/novartis/nct_numbers")
async def get_nct_number():
    conn = get_db_connection()
    if conn is None:
        raise HTTPException(status_code=503, detail="Database connection failed.")
    try:
        # Query the database to get distinct NCT numbers
        query = "SELECT DISTINCT(NCT_Number) FROM embedding"
        df_saved = pd.read_sql(query, conn)
        nct_to_drop = "NCT00196794, NCT00002273, NCT03757065, NCT01931644, NCT05635266, NCT04885920, NCT00810030, NCT03482648, NCT05596422, NCT04469686, NCT00450086, NCT00466635, NCT01149694, NCT00034294, NCT06025227, NCT05051943, NCT01278082, NCT00577473, NCT02423460, NCT05844592, NCT00382304, NCT02472769, NCT00621192, NCT03835780, NCT04872491, NCT05579444, NCT03599622, NCT03090139, NCT01813526, NCT03488030, NCT03327558, NCT01630096, NCT01301417, NCT01954017, NCT02497417, NCT03705117, NCT03251118, NCT04003818, NCT01369329, NCT03452501, NCT05999708, NCT02073214, NCT00663819, NCT05913817, NCT03662919, NCT03143517, NCT01326013, NCT01880918, NCT00977912, NCT03836612, NCT01476995, NCT02674308, NCT06246123, NCT03071081, NCT02493712, NCT01100112, NCT04089514, NCT00606346, NCT05535946, NCT05092269, NCT02306772, NCT02539368, NCT00917514, NCT02925338, NCT05771155, NCT01369355, NCT05633706, NCT04641442, NCT04018599, NCT05831670, NCT01290042, NCT01287559, NCT05463900, NCT03978000, NCT02678052, NCT04456400, NCT03532932, NCT02306785, NCT02760615, NCT05427942, NCT05190484, NCT05598684, NCT03952364, NCT02368743, NCT04239521, NCT00618202, NCT04360343, NCT00072943, NCT06315738, NCT00867438, NCT01209208, NCT05626088, NCT03329209, NCT00106509, NCT02087878, NCT01058356, NCT02209987, NCT02750800, NCT05200000, NCT02299570, NCT03597971, NCT05636709, NCT00505778, NCT05807971, NCT03491228, NCT06206811, NCT06363383, NCT04597905, NCT04458805, NCT03801928, NCT02306798, NCT02246361, NCT03914261, NCT05121753, NCT05659953, NCT00603733, NCT04959851, NCT02629211, NCT03011593, NCT00618228, NCT04567628, NCT03038711, NCT04995783, NCT03006198, NCT01984047, NCT03609294, NCT06297291, NCT03858894, NCT00354991, NCT02786082, NCT03662620, NCT05335044, NCT02345044, NCT00956644, NCT05195827, NCT05332873, NCT02194465, NCT03726866, NCT04200573, NCT00335673, NCT01848873, NCT03742544, NCT04811092, NCT04258865, NCT01557582, NCT00553865, NCT02840565, NCT03884465, NCT05131074, NCT00821574, NCT03556020, NCT03888365, NCT05205265, NCT00518765, NCT00824720, NCT02807974, NCT00041574, NCT00739674, NCT01102465, NCT05198674, NCT00855465, NCT04162366, NCT00785057, NCT03099226, NCT00539526, NCT01797926, NCT05934526, NCT04835857, NCT01251848, NCT02679248, NCT05259020, NCT00946465, NCT05712226, NCT01712126, NCT04668157, NCT03009474, NCT01244620, NCT01337674, NCT01222520, NCT05500820, NCT00618774, NCT06268873, NCT05043831, NCT01117831, NCT02242331, NCT00385931, NCT02811731, NCT03231982, NCT00171574, NCT04813926, NCT00796666, NCT04714320, NCT00556920, NCT02995720, NCT01271374, NCT00185120, NCT01297920, NCT00837720, NCT02246348, NCT05725148, NCT04903730, NCT00637130, NCT06052748, NCT03686657, NCT01150357, NCT04041648, NCT01025791, NCT02847260, NCT00160160, NCT01096160, NCT00373360, NCT01557660, NCT01287260, NCT00768560, NCT01292460, NCT04945460, NCT04669548, NCT06048120, NCT01934582, NCT00865020, NCT00708422, NCT04083222, NCT00230763, NCT04373863, NCT00170963, NCT01210963, NCT02030886, NCT00470886, NCT03935386, NCT04095286, NCT04311086, NCT00168857, NCT00467896, NCT00147563, NCT01033422, NCT00110422, NCT01864863, NCT05097794, NCT00841880, NCT01911780, NCT05040880, NCT06484673, NCT00157729, NCT01274494, NCT01625494, NCT04991194, NCT02773862, NCT00892762, NCT01799473, NCT00200473, NCT05928676, NCT03865576, NCT03714776, NCT00767494, NCT01342094, NCT00537043, NCT04035538, NCT01768403, NCT02126943, NCT02211638, NCT00338338, NCT01582438, NCT04967443, NCT01033643, NCT00274638, NCT01016691, NCT01050062, NCT00185094, NCT04030494, NCT04158076, NCT01690676, NCT00219076, NCT00865176, NCT00200460, NCT00702260, NCT04994860, NCT01426594, NCT01583582, NCT01340131, NCT01332331, NCT01406431, NCT00963027, NCT06226727, NCT01493427, NCT03503773, NCT02017327, NCT02984631, NCT01415531, NCT00297973, NCT06220773, NCT01648231, NCT01056731, NCT01165476, NCT02151994, NCT00244660, NCT06348576, NCT01393860, NCT00347360, NCT00307060, NCT05107960, NCT01105104, NCT03708146, NCT05838872, NCT00643604, NCT02725372, NCT00902304, NCT01132768, NCT01722604, NCT00862472, NCT00819104, NCT00383929, NCT00946829, NCT01352689, NCT02007629, NCT03045029, NCT00800592, NCT02436512, NCT05938946, NCT00530946, NCT02041104, NCT01743001, NCT02773901, NCT03750201, NCT00170989, NCT04207229, NCT00680329, NCT02248129, NCT02822729, NCT00143429, NCT01896180, NCT01077180, NCT00877929, NCT01355380, NCT02652429, NCT03012646, NCT02384772, NCT01895972, NCT03090568, NCT06016972, NCT00241072, NCT01426867, NCT00434967, NCT00335244, NCT04061044, NCT01602367, NCT01998867, NCT01263444, NCT02006589, NCT03450629, NCT01042392, NCT01699529, NCT01306292, NCT04694989, NCT00934089, NCT00219089, NCT02222480, NCT00953680, NCT02988362, NCT00219180, NCT05372380, NCT00093262, NCT00392262, NCT00250562, NCT02951962, NCT00702091, NCT05843162, NCT06448962, NCT05857267, NCT02087540, NCT02998840, NCT00708344, NCT00171067, NCT01897740, NCT00891267, NCT06129240, NCT02548286, NCT01252563, NCT01258673, NCT00348686, NCT01822639, NCT01956786, NCT06036186, NCT01712139, NCT06501586, NCT00930722, NCT02172586, NCT01664039, NCT05593562, NCT05703191, NCT01227291, NCT05077462, NCT00435162, NCT00596791, NCT02003391, NCT00240422, NCT01071122, NCT01236339, NCT01872039, NCT03519386, NCT00442286, NCT01806363, NCT03683186, NCT01289886, NCT05660863, NCT00219063, NCT00546286, NCT00573430, NCT01409122, NCT05470725, NCT02092025, NCT05808725, NCT00734630, NCT01277822, NCT06429930, NCT00760539, NCT01853839, NCT01066039, NCT01658839, NCT00386139, NCT01153386, NCT00795639, NCT03810586, NCT00082186, NCT03950739, NCT01361139, NCT00367939, NCT05413057, NCT02801526, NCT01913457, NCT05236348, NCT04470830, NCT05257148, NCT03463148, NCT05211648, NCT05462535, NCT00234858, NCT00642096, NCT01918358, NCT02248896, NCT02933658, NCT00362258, NCT02829996, NCT01076296, NCT03031496, NCT00982735, NCT02791958, NCT01474135, NCT02242396, NCT04006158, NCT05612035, NCT05660135, NCT05921435, NCT02348658, NCT01755858, NCT01286558, NCT00791258, NCT02006758, NCT03217825, NCT01508325, NCT01842230, NCT00946725, NCT06428825, NCT05370625, NCT06242535, NCT01835535, NCT03122730, NCT06441630, NCT01108796, NCT02186496, NCT01172496, NCT00508365, NCT05369065, NCT00624065, NCT00950066, NCT02927366, NCT00995566, NCT03960866, NCT01266265, NCT00670566, NCT00913965, NCT06121518, NCT00244634, NCT01297517, NCT03783117, NCT00602017, NCT00878917, NCT01830517, NCT04223934, NCT05667818, NCT00296218, NCT01789918, NCT06061718, NCT04466501, NCT05389267, NCT03247140, NCT00882440, NCT00219167, NCT01871740, NCT01623492, NCT04930289, NCT03439189, NCT02260089, NCT05731492, NCT04161001, NCT01415401, NCT03399604, NCT05549401, NCT06074601, NCT03648801, NCT00963001, NCT00414869, NCT01456169, NCT01469169, NCT01243268, NCT01496469, NCT00938262, NCT02655029, NCT05117762, NCT04592380, NCT05427162, NCT03401580, NCT03639480, NCT03825380, NCT00109681, NCT01667510, NCT02496910, NCT05780710, NCT04675944, NCT01975246, NCT02909868, NCT02955368, NCT03632668, NCT03016468, NCT01876368, NCT00386529, NCT00799604, NCT06355544, NCT03364244, NCT00274144, NCT00171444, NCT01355367, NCT03800901, NCT00683501, NCT02594410, NCT03624010, NCT00235014, NCT02780414, NCT01151410, NCT03762369, NCT02152969, NCT06274801, NCT05223101, NCT03131167, NCT00239538, NCT00051168, NCT01420068, NCT00146341, NCT00646841, NCT00554619, NCT00490841, NCT04994119, NCT02387619, NCT00834041, NCT01265719, NCT00102141, NCT04947124, NCT00143234, NCT02181816, NCT02777216, NCT00170924, NCT02094924, NCT02358824, NCT03653624, NCT00844324, NCT02630316, NCT03397524, NCT05279716, NCT00241124, NCT03116516, NCT04667624, NCT00440141, NCT00108017, NCT00398541, NCT00679224, NCT00496834, NCT05746117, NCT01833741, NCT06508619, NCT02325518, NCT01183741, NCT06144918, NCT01922141, NCT00171119, NCT00761319, NCT01179334, NCT02075619, NCT00274118, NCT06227819, NCT00433836, NCT00325936, NCT00848536, NCT04524416, NCT01976624, NCT06180096, NCT00891124, NCT01560624, NCT04883658, NCT01251835, NCT01907958, NCT05017935, NCT00157963, NCT00882336, NCT00543413, NCT00089713, NCT01562613, NCT01033318, NCT01302249, NCT00798759, NCT05641675, NCT05077475, NCT01704170, NCT01954446, NCT01590810, NCT03891446, NCT00443612, NCT01357746, NCT00918346, NCT02874846, NCT00777946, NCT00439946, NCT01079195, NCT03602781, NCT03901781, NCT04991181, NCT00648895, NCT05161481, NCT00767481, NCT01342081, NCT01365481, NCT03461081, NCT01476995, NCT03992755, NCT04991155, NCT04589923, NCT04448223, NCT02343250, NCT02419508, NCT05983250, NCT00260923, NCT02001350, NCT00887250, NCT00244595, NCT06475781, NCT00045981, NCT04647214, NCT00051181, NCT03748212, NCT00403481, NCT00659581, NCT05540912, NCT01074281, NCT00662610, NCT00334581, NCT00549510, NCT05622695, NCT04284514, NCT00613314, NCT01001195, NCT01831895, NCT00151814, NCT05552495, NCT01937312, NCT00311012, NCT05181046, NCT00785512, NCT00274612, NCT02250612, NCT02278614, NCT00457795, NCT03254914, NCT01263314, NCT05834595, NCT02042014, NCT00242814, NCT05867914, NCT02358369, NCT06184269, NCT00542269, NCT01921946, NCT02981446, NCT00508469, NCT00294710, NCT03649646, NCT05495269, NCT03657472, NCT01529372, NCT01656408, NCT04521023, NCT03657550, NCT00143208, NCT03833323, NCT06104423, NCT00153023, NCT03017950, NCT03246555, NCT05208814, NCT01013155, NCT00370214, NCT04585555, NCT00402103, NCT03128138, NCT06059638, NCT00552708, NCT01453855, NCT02569814, NCT00241150, NCT00394823, NCT01847014, NCT00572455, NCT02910414, NCT01340014, NCT03920579, NCT02187484, NCT02621008, NCT02053623, NCT00615108, NCT03438123, NCT05086523, NCT05526690, NCT02155790, NCT04565990, NCT01005290, NCT02459990, NCT02891850, NCT06165250, NCT00803634, NCT03730116, NCT05966324, NCT00147524, NCT03090724, NCT03566316, NCT01392534, NCT05931224, NCT01138124, NCT01806324, NCT05651724, NCT02495324, NCT03868124, NCT00937534, NCT01190436, NCT00412113, NCT01995136, NCT01789736, NCT01037036, NCT00865618, NCT01798849, NCT01534299, NCT04149899, NCT00140049, NCT03847506, NCT00828906, NCT01327599, NCT06368206, NCT00961649, NCT05571813, NCT02032836, NCT00093249, NCT01289899, NCT05086549, NCT03609606, NCT01937299, NCT01908699, NCT05491642, NCT01587742, NCT01974570, NCT02152306, NCT00772499, NCT00576342, NCT00200499, NCT02344199, NCT01622842, NCT00638742, NCT02939599, NCT04587206, NCT00883506, NCT05450575, NCT03014375, NCT00151775, NCT00749775, NCT02526875, NCT02057575, NCT06513975, NCT03707899, NCT01819870, NCT00274599, NCT06120842, NCT00573742, NCT00802542, NCT02822742, NCT05440513, NCT01712113, NCT05988099, NCT00750113, NCT05993806, NCT02024100, NCT00140959, NCT04807959, NCT02504606, NCT02439749, NCT00595049, NCT00047606, NCT05397106, NCT02178306, NCT05899959, NCT00796159, NCT05875259, NCT00943852, NCT04676152, NCT06281132, NCT04688632, NCT06214832, NCT02746237, NCT06350032, NCT05614037, NCT00922532, NCT05286632, NCT03872232, NCT00256152, NCT01176032, NCT00171132, NCT02827032, NCT01342952, NCT00938132, NCT01318252, NCT03067415, NCT01390415, NCT02262637, NCT02565615, NCT00781885, NCT06009185, NCT05027685, NCT03744637, NCT02357615, NCT00219115, NCT00133185, NCT00794885, NCT04388215, NCT01338415, NCT00171015, NCT01873885, NCT05060315, NCT00185185, NCT00620256, NCT03586037, NCT06249152, NCT00546052, NCT00299832, NCT06008015, NCT03556085, NCT02413515, NCT02537015, NCT04069715, NCT05397600, NCT04019652, NCT00171600, NCT01216852, NCT03250052, NCT01987752, NCT01426100, NCT00458042, NCT00716742, NCT02608242, NCT00170937, NCT01217879, NCT00150384, NCT05184179, NCT00526279, NCT02385721, NCT03464864, NCT00258921, NCT01295021, NCT01652664, NCT05632302, NCT00244621, NCT01028664, NCT02006602, NCT00558064, NCT00946621, NCT01482364, NCT01107743, NCT04316143, NCT05698043, NCT01227603, NCT05460364, NCT02397590, NCT04673864, NCT00283764, NCT04185090, NCT05631990, NCT00673790, NCT01631864, NCT00267943, NCT01587638, NCT00990743, NCT01070043, NCT00626743, NCT01248338, NCT04019743, NCT00409643, NCT05469503, NCT03541603, NCT06441643, NCT01446003, NCT02242864, NCT02695264, NCT04668664, NCT00883064, NCT00441064, NCT05282121, NCT01699464, NCT04463121, NCT00561964, NCT00789321, NCT00171002, NCT02235402, NCT00541684, NCT01078584, NCT06325384, NCT00255502, NCT00549302, NCT02382484, NCT05245084, NCT02064621, NCT00219102, NCT01731002, NCT01243138, NCT00350038, NCT02397538, NCT04896008, NCT00439738, NCT06082843, NCT01146938, NCT04747808, NCT00950690, NCT03278002, NCT02704702, NCT00601302, NCT00821002, NCT02337790, NCT01026103, NCT02205190, NCT03767803, NCT02143843, NCT05526703, NCT00902603, NCT04467879, NCT01335984, NCT01330979, NCT01900184, NCT04600284, NCT05961384, NCT05372679, NCT00949884, NCT01654484, NCT03849287, NCT00942487, NCT01593787, NCT01108809, NCT02989909, NCT00171756, NCT05192356, NCT04708756, NCT02298556, NCT05881707, NCT01041807, NCT01190007, NCT04654507, NCT01265888, NCT00331188, NCT00878878, NCT01102478, NCT03655288, NCT03197688, NCT03103256, NCT01918709, NCT01721707, NCT01389609, NCT02336607, NCT01350609, NCT06291207, NCT02108288, NCT01764178, NCT00939588, NCT03315507, NCT00140907, NCT01200407, NCT01447485, NCT02821156, NCT00233532, NCT04796337, NCT00744237, NCT06072937, NCT04505137, NCT01842256, NCT01031485, NCT02579356, NCT01363336, NCT03659149, NCT00047554, NCT04024293, NCT03255993, NCT00980187, NCT00219154, NCT00549133, NCT05977933, NCT02282033, NCT02250833, NCT02112487, NCT05411887, NCT01196533, NCT02789475, NCT05385770, NCT01092559, NCT02439775, NCT02400775, NCT02047175, NCT02200575, NCT06259175, NCT05451875, NCT01254370, NCT04981470, NCT00673075, NCT01080742, NCT04660370, NCT03827200, NCT03179800, NCT01712100, NCT01739400, NCT02651870, NCT04987970, NCT00362037, NCT00676637, NCT02845037, NCT05513937, NCT01510132, NCT03205137, NCT01844037, NCT06518915, NCT01294215, NCT06099015, NCT05459688, NCT02773888, NCT01736488, NCT04488978, NCT00426478, NCT00367978, NCT00234871, NCT04991207, NCT02231788, NCT06112678, NCT02064556, NCT00396656, NCT06063109, NCT01452009, NCT01609907, NCT05279807, NCT05963009, NCT01101009, NCT02914509, NCT00360178, NCT02587988, NCT04084678, NCT03645278, NCT02772471, NCT01216878, NCT03626688, NCT00620178, NCT00904371, NCT01087671, NCT01211171, NCT01369771, NCT01900171, NCT01646671, NCT00561171, NCT02730871, NCT00219193, NCT00871871, NCT01246193, NCT00158093, NCT03198793, NCT05185011, NCT00879411, NCT00440011, NCT01134393, NCT04917393, NCT00834171, NCT00700271, NCT04398771, NCT06183671, NCT00557128, NCT00770861, NCT03231293, NCT00539487, NCT02688387, NCT02807987, NCT02060487, NCT01298687, NCT00709787, NCT01528787, NCT04898387, NCT02448628, NCT01114893, NCT00441883, NCT00866983, NCT01206361, NCT01306461, NCT00967811, NCT01062971, NCT00154271, NCT01806311, NCT01659411, NCT03697811, NCT00991783, NCT01458483, NCT00944983, NCT05961397, NCT00847483, NCT01362283, NCT01077661, NCT02016183, NCT01278797, NCT01320397, NCT00770497, NCT00171405, NCT00991705, NCT00274105, NCT00938197, NCT01907828, NCT06008028, NCT05247528, NCT02633293, NCT00449111, NCT01928628, NCT02428998, NCT00241098, NCT02612298, NCT06317805, NCT01588405, NCT05462405, NCT00332761, NCT00272961, NCT00422461, NCT01814761, NCT05878561, NCT05579561, NCT01975961, NCT01370005, NCT00480805, NCT03489005, NCT03043651, NCT00679653, NCT02190877, NCT01501253, NCT01557647, NCT06356077, NCT00550953, NCT05645653, NCT00368277, NCT00159653, NCT05765253, NCT03249753, NCT00491777, NCT00171353, NCT01536353, NCT03727451, NCT04120753, NCT02205151, NCT01241487, NCT02387554, NCT01012687, NCT00912054, NCT00923533, NCT04891354, NCT03896633, NCT02804087, NCT00937651, NCT01926951, NCT02242851, NCT00564187, NCT06266351, NCT01065051, NCT00751751, NCT05427253, NCT02566187, NCT00047593, NCT02586311, NCT00626028, NCT00284128, NCT05892328, NCT02761811, NCT00219128, NCT00255528, NCT00869193, NCT02262611, NCT00713011, NCT03795428, NCT02187497, NCT03371797, NCT01696383, NCT01911897, NCT00327145, NCT00394745, NCT00654745, NCT02062645, NCT01104545, NCT02994745, NCT04609345, NCT04656847, NCT02278445, NCT03884647, NCT05347147, NCT01510145, NCT00767247, NCT02003547, NCT04075045, NCT00960245, NCT02837445, NCT02444845, NCT00277498, NCT01131845, NCT04312698, NCT00676845, NCT00160498, NCT00666198, NCT02304198, NCT01547598, NCT03536598, NCT00617877, NCT01167153, NCT00160277, NCT01113047, NCT06431477, NCT00814645, NCT02920047, NCT05363761, NCT00159861, NCT02770261, NCT05749861, NCT06235554, NCT00957554, NCT05356754, NCT01312454, NCT01429233, NCT04691154, NCT05719454, NCT03088254, NCT00579254, NCT02856633, NCT01663233, NCT03549754, NCT00056524, NCT00095719, NCT00151502, NCT00165659, NCT00165750, NCT00236431, NCT00236574, NCT00249145, NCT00249158, NCT00253123, NCT00261573, NCT00366288, NCT00366483, NCT00403520, NCT00452504, NCT00468897, NCT00474552, NCT00479297, NCT00479349, NCT00479440, NCT00479700, NCT00479843, NCT00480467, NCT00480818, NCT00483028, NCT00494962, NCT00499200, NCT00499642, NCT00551772, NCT00563732, NCT00582127, NCT00582855, NCT00621647, NCT00624026, NCT00649220, NCT00660088, NCT00684710, NCT00687141, NCT00689559, NCT00689637, NCT00692510, NCT00711139, NCT00711321, NCT00713765, NCT00718731, NCT00719394, NCT00726115, NCT00733785, NCT00742417, NCT00745576, NCT00765115, NCT00777361, NCT00788047, NCT00795730, NCT00814346, NCT00824590, NCT00825084, NCT00827034, NCT00829816, NCT00831506, NCT00838084, NCT00838877, NCT00843115, NCT00857415, NCT00857506, NCT00860275, NCT00867399, NCT00884533, NCT00889603, NCT00906191, NCT00916617, NCT00931073, NCT00954252, NCT00954369, NCT00954538, NCT00975481, NCT00979316, NCT00987220, NCT00988624, NCT00990613, NCT00991419, NCT01002079, NCT01020838, NCT01024660, NCT01039194, NCT01042314, NCT01057030, NCT01072812, NCT01079819, NCT01093664, NCT01133405, NCT01137799, NCT01138111, NCT01152216, NCT01192529, NCT01203384, NCT01211782, NCT01221259, NCT01227252, NCT01230853, NCT01253122, NCT01253499, NCT01258452, NCT01294540, NCT01303744, NCT01309763, NCT01325662, NCT01348737, NCT01366027, NCT01370954, NCT01378195, NCT01385033, NCT01421056, NCT01447719, NCT01454115, NCT01465360, NCT01467726, NCT01503944, NCT01518374, NCT01537757, NCT01550549, NCT01564706, NCT01565356, NCT01565369, NCT01568086, NCT01592552, NCT01602393, NCT01614886, NCT01658722, NCT01660815, NCT01672827, NCT01702467, NCT01702480, NCT01703702, NCT01716897, NCT01723488, NCT01723670, NCT01733355, NCT01741194, NCT01745198, NCT01764243, NCT01860625, NCT01886820, NCT01924858, NCT01946243, NCT01978327, NCT01992380, NCT02016560, NCT02029547, NCT02040987, NCT02051335, NCT02051764, NCT02051790, NCT02061878, NCT02078310, NCT02107599, NCT02114372, NCT02120664, NCT02130661, NCT02178124, NCT02278354, NCT02278367, NCT02291783, NCT02333942, NCT02336360, NCT02340195, NCT02350634, NCT02489110, NCT02516046, NCT02534480, NCT02537938, NCT02546310, NCT02562989, NCT02576639, NCT02621606, NCT02648672, NCT02681172, NCT02695004, NCT02710188, NCT02778438, NCT02778581, NCT02782975, NCT02795052, NCT02795780, NCT02813070, NCT02840279, NCT02843529, NCT02859207, NCT02860338, NCT02875496, NCT02910102, NCT02910739, NCT02928211, NCT02968719, NCT02973971, NCT03030105, NCT03030586, NCT03069014, NCT03089918, NCT03172117, NCT03226522, NCT03259958, NCT03298672, NCT03308032, NCT03322462, NCT03328195, NCT03371420, NCT03387267, NCT03418688, NCT03432195, NCT03438604, NCT03456349, NCT03461276, NCT03467477, NCT03484143, NCT03493282, NCT03531684, NCT03531710, NCT03538522, NCT03551769, NCT03556280, NCT03577262, NCT03587376, NCT03635047, NCT03635879, NCT03661034, NCT03685240, NCT03698695, NCT03711825, NCT03770182, NCT03784300, NCT03802162, NCT03811184, NCT03822208, NCT03838185, NCT03865017, NCT03884647, NCT03899298, NCT03901092, NCT03919162, NCT03935568, NCT03971123, NCT04023994, NCT04074837, NCT04114994, NCT04157712, NCT04251182, NCT04268953, NCT04311515, NCT04314934, NCT04384978, NCT04394845, NCT04413851, NCT04449926, NCT04462029, NCT04474405, NCT04476303, NCT04489303, NCT04498650, NCT04517877, NCT04559789, NCT04585347, NCT04672135, NCT04715750, NCT04745104, NCT04770220, NCT04897464, NCT04920786, NCT04931459, NCT04939818, NCT04973189, NCT04983368, NCT05009524, NCT05028114, NCT05058040, NCT05077501, NCT05077631, NCT05107882, NCT05153161, NCT05153941, NCT05181475, NCT05215782, NCT05225389, NCT05231785, NCT05248672, NCT05321498, NCT05344989, NCT05345509, NCT05352529, NCT05395624, NCT05406778, NCT05408780, NCT05418296, NCT05422339, NCT05477056, NCT05482867, NCT05503511, NCT05515679, NCT05516134, NCT05516147, NCT05516342, NCT05525780, NCT05527405, NCT05529706, NCT05542953, NCT05575076, NCT05628636, NCT05635266, NCT05641688, NCT05686044, NCT05696483, NCT05783830, NCT05796037, NCT05817643, NCT05888610, NCT05892952, NCT05916664, NCT05921929, NCT05959239, NCT05965414, NCT05986721, NCT06025877, NCT06043700, NCT06079216, NCT06127368, NCT06151795, NCT06151808, NCT06194552, NCT06217146, NCT06223438, NCT06234930, NCT06247345, NCT06298474, NCT06304883, NCT06367426, NCT06373094, NCT06388551, NCT06390098, NCT06406348, NCT06412185, NCT06511570"  # Replace with your list
        nct_list = [n.strip() for n in nct_to_drop.split(',')]

        # Drop rows where 'NCT Number' matches any value in nct_list
        df_filtered = df_saved[~df_saved['NCT_Number'].isin(nct_list)]
        distinct_nct_numbers = df_filtered['NCT_Number'].tolist()
        distinct_nct_numbers.append("Not Available")

        return JSONResponse(content={"nctNumbers": distinct_nct_numbers})
    finally:
        conn.close()

# Endpoint to fetch trial details based on NCT number
@app.post("/api/novartis/trial_details")
async def get_trial_details(request: Request):
    payload = await request.json()  # Parse the incoming JSON payload
    nctNumber = payload.get("nctNumber")

    if not nctNumber or nctNumber.lower() == "not available":
        # Return empty trial details if NCT number is not available
        empty_trial_details = {
            "studyTitle": "",
            "primaryOutcomeMeasures": "",
            "secondaryOutcomeMeasures": "",
            "inclusionCriteria": "",
            "exclusionCriteria": ""
        }
        return JSONResponse(content={"trialDetails": [empty_trial_details]})

    conn = get_db_connection()
    if conn is None:
        raise HTTPException(status_code=503, detail="Database connection failed.")

    try:
        # Query for trial details by NCT number
        query = """
            SELECT 
                Study_Title, 
                Primary_Outcome_Measures, 
                Secondary_Outcome_Measures, 
                Inclusion_Criteria, 
                Exclusion_Criteria 
            FROM clinicaltrials 
            WHERE LOWER(NCT_Number) = %s
        """
        df_saved = pd.read_sql(query, conn, params=(nctNumber.lower(),))

        # Convert database rows to a list of dictionaries and camelCase keys
        trial_details = df_saved.to_dict(orient='records')
        trial_details_camel_case = [
            {
                "studyTitle": record["Study_Title"],
                "primaryOutcomeMeasures": record["Primary_Outcome_Measures"],
                "secondaryOutcomeMeasures": record["Secondary_Outcome_Measures"],
                "inclusionCriteria": record["Inclusion_Criteria"],
                "exclusionCriteria": record["Exclusion_Criteria"]
            }
            for record in trial_details
        ]

        return JSONResponse(content={"trialDetails": trial_details_camel_case})
    finally:
        conn.close()

# Endpoint to get top trials based on various parameters
@app.post("/api/novartis/top_trials")
async def get_top_trials(request: Request):
    payload = await request.json()  # Parse the incoming JSON payload

    # Extract parameters from payload
    nctNumber = payload.get("nctCode")
    studyTitle = payload.get("studyTitle")
    primaryOutcomeMeasures = payload.get("primaryOutcome")
    secondaryOutcomeMeasures = payload.get("secondaryOutcome")
    inclusionCriteria = payload.get("inclusionCriteria")
    exclusionCriteria = payload.get("exclusionCriteria")

    # Ensure at least one argument is provided and not blank
    if all(not arg for arg in [
        studyTitle,
        primaryOutcomeMeasures,
        secondaryOutcomeMeasures,
        inclusionCriteria,
        exclusionCriteria
    ]):
        raise HTTPException(status_code=400, detail="At least one argument must be provided and not blank.")
    # Call the trials_extraction function to get trial data
    try:
        result = trials_extraction(
            nctNumber,
            studyTitle,
            primaryOutcomeMeasures,
            secondaryOutcomeMeasures,
            inclusionCriteria,
            exclusionCriteria
        )
        # Check if result is a string (error message from the model)
        if isinstance(result, str):
            raise HTTPException(status_code=400, detail="The model is trained on Ulcerative Colitis, Hypertension, and Alzheimer. Please provide relevant data for these diseases.")

        # Ensure result is a DataFrame
        if not isinstance(result, pd.DataFrame):
            raise ValueError("Unexpected result type from trials_extraction. Expected DataFrame.")

        conn = get_db_connection()
        if conn is None:
            raise HTTPException(status_code=503, detail="Database connection failed.")

        try:
            # Rename columns in the DataFrame for consistency
            result.columns = [
                "nctNumber", "studyTitle", "primaryOutcomeMeasures", "secondaryOutcomeMeasures",
                "inclusionCriteria", "exclusionCriteria", "disease", "drug", "drugSimilarity",
                "inclusionCriteriaSimilarity", "exclusionCriteriaSimilarity",
                "studyTitleSimilarity", "primaryOutcomeMeasuresSimilarity",
                "secondaryOutcomeMeasuresSimilarity", "overallSimilarity"
            ]

            # Convert the DataFrame to a list of dictionaries
            trials_list = result.to_dict(orient="records")

            # Insert the response into the database
            insert_db(
                nctNumber, studyTitle, primaryOutcomeMeasures, secondaryOutcomeMeasures,
                inclusionCriteria, exclusionCriteria, json.dumps(trials_list), conn
            )

            # Filter the result for a specific set of fields
            filtered_trials_list = [
                {
                    "nctNumber": trial["nctNumber"],
                    "studyTitle": trial["studyTitle"],
                    "overallSimilarity": trial["overallSimilarity"]
                }
                for trial in trials_list
            ]

            return JSONResponse(content={"trials": filtered_trials_list})
        finally:
            conn.close()

    except ValueError as e:
        return JSONResponse(
            content={"message": str(e)},
            status_code=400
        )

    except Exception as e:
        return JSONResponse(
            content={"message": "An unexpected error occurred. Please try again later.", "error": str(e)},
            status_code=500
        )

# Endpoint to fetch a specific trial based on NCT number from history data
@app.post("/api/novartis/particular_trial")
async def get_particular_trial(request: Request):
    payload = await request.json()  # Parse the incoming JSON payload

    nctNumber = payload.get("nctNumber")
    if not nctNumber:
        raise HTTPException(status_code=400, detail="nctNumber is required.")

    conn = get_db_connection()
    if conn is None:
        raise HTTPException(status_code=503, detail="Database connection failed.")

    try:
        # Query to fetch the latest trial data from history
        query = """
                SELECT response
                FROM history
                WHERE Serial_Number = (SELECT MAX(Serial_Number) FROM history);
        """
        df_saved = pd.read_sql(query, conn)

        if df_saved.empty:
            raise HTTPException(status_code=404, detail="No trial data found.")

        response_data = df_saved['response'].iloc[0]  # Get most recent response

        # Deserialize JSON string into a Python object
        trials_list = json.loads(response_data)

        # Ensure the trials list is valid
        if not isinstance(trials_list, list):
            raise HTTPException(status_code=500, detail="Response data is not a valid list of trials.")

        # Filter trials matching the provided NCT number
        filtered_trials = [
            trial for trial in trials_list if trial.get('nctNumber', '').lower() == nctNumber.lower()
        ]

        if not filtered_trials:
            raise HTTPException(status_code=404, detail="Trial not found for the given NCT number.")

        # Return the filtered trial data
        result = [
            {
                "nctNumber": trial["nctNumber"],
                "disease": trial.get("disease", "Not Available"),
                "studyTitle": trial.get("studyTitle", "Not Available"),
                "primaryOutcomeMeasures": trial.get("primaryOutcomeMeasures", "Not Available"),
                "secondaryOutcomeMeasures": trial.get("secondaryOutcomeMeasures", "Not Available"),
                "inclusionCriteria": trial.get("inclusionCriteria", "Not Available"),
                "exclusionCriteria": trial.get("exclusionCriteria", "Not Available"),
                "drug": trial.get("drug", "Not Available"),
                "studyTitleSimilarity": trial.get("studyTitleSimilarity", 0.0),
                "primaryOutcomeMeasuresSimilarity": trial.get("primaryOutcomeMeasuresSimilarity", 0.0),
                "secondaryOutcomeMeasuresSimilarity": trial.get("secondaryOutcomeMeasuresSimilarity", 0.0),
                "inclusionCriteriaSimilarity": trial.get("inclusionCriteriaSimilarity", 0.0),
                "exclusionCriteriaSimilarity": trial.get("exclusionCriteriaSimilarity", 0.0),
                "drugSimilarity": trial.get("drugSimilarity", 0.0),
                "overallSimilarity": trial.get("overallSimilarity", 0.0)
            }
            for trial in filtered_trials
        ]

        return JSONResponse(content={"trialDetails": result})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

    finally:
        if conn:
            conn.close()

# Endpoint to fetch history of trial inputs from the database
@app.get("/api/novartis/input_history")
async def get_history_input():
    conn = get_db_connection()
    if conn is None:
        raise HTTPException(status_code=503, detail="Database connection failed.")

    try:
        # Query to fetch the latest trial data from the history table
        query = """
                SELECT NCT_Number,
                Study_Title, 
                Primary_Outcome_Measures, 
                Secondary_Outcome_Measures, 
                Inclusion_Criteria, 
                Exclusion_Criteria 
                FROM history
                WHERE Serial_Number = (SELECT MAX(Serial_Number) FROM history);
        """
        df_saved = pd.read_sql(query, conn)

        if df_saved.empty:
            raise HTTPException(status_code=404, detail="No trial data found.")

        trial_details = df_saved.to_dict(orient='records')
        trial_details_camel_case = [
            {
                "nctNumber": record["NCT_Number"],
                "studyTitle": record["Study_Title"],
                "primaryOutcomeMeasures": record["Primary_Outcome_Measures"],
                "secondaryOutcomeMeasures": record["Secondary_Outcome_Measures"],
                "inclusionCriteria": record["Inclusion_Criteria"],
                "exclusionCriteria": record["Exclusion_Criteria"]
            }
            for record in trial_details
        ]

        return JSONResponse(content={"trialDetails": trial_details_camel_case})
    finally:
        conn.close()

# Endpoint to fetch top trials for passing nct number only
@app.post("/api/novartis/top_trials_nct")
async def get_top_trials(request: Request):
    try:
        # Parse incoming JSON payload
        payload = await request.json()
        nctNumber = payload.get("nctNumber")

        # Handle case where NCT number is not provided or invalid
        if not nctNumber or nctNumber.lower() == "not available":
            empty_trial_details = {
                "studyTitle": "",
                "primaryOutcomeMeasures": "",
                "secondaryOutcomeMeasures": "",
                "inclusionCriteria": "",
                "exclusionCriteria": ""
            }
            return JSONResponse(content={"trialDetails": [empty_trial_details]}, status_code=200)

        # Establish database connection
        conn = get_db_connection()
        if conn is None:
            raise HTTPException(status_code=503, detail="Database connection failed.")

        # Fetch trial details based on the NCT number
        query = """
            SELECT 
                Study_Title, 
                Primary_Outcome_Measures, 
                Secondary_Outcome_Measures, 
                Inclusion_Criteria, 
                Exclusion_Criteria 
            FROM clinicaltrials 
            WHERE LOWER(NCT_Number) = %s
        """
        df_saved = pd.read_sql(query, conn, params=(nctNumber.lower(),))
        if df_saved.empty:
            raise HTTPException(status_code=404, detail="No trial found for the provided NCT number.")

        # Extract details
        studyTitle = df_saved["Study_Title"].iloc[0]
        primaryOutcomeMeasures = df_saved["Primary_Outcome_Measures"].iloc[0]
        secondaryOutcomeMeasures = df_saved["Secondary_Outcome_Measures"].iloc[0]
        inclusionCriteria = df_saved["Inclusion_Criteria"].iloc[0]
        exclusionCriteria = df_saved["Exclusion_Criteria"].iloc[0]

        # Call trials_extraction function to process the data
        result = trials_extraction(
            nctNumber,
            studyTitle,
            primaryOutcomeMeasures,
            secondaryOutcomeMeasures,
            inclusionCriteria,
            exclusionCriteria
        )

        # Handle error or limitation messages from the trials_extraction function
        if isinstance(result, str):
            return JSONResponse(content={"message": result}, status_code=200)

        # Validate result format
        if not isinstance(result, pd.DataFrame):
            raise ValueError("Unexpected result type from trials_extraction. Expected a DataFrame.")

        # Rename columns for consistency
        result.columns = [
            "nctNumber", "studyTitle", "primaryOutcomeMeasures", "secondaryOutcomeMeasures",
            "inclusionCriteria", "exclusionCriteria", "disease", "drug", "drugSimilarity",
            "inclusionCriteriaSimilarity", "exclusionCriteriaSimilarity",
            "studyTitleSimilarity", "primaryOutcomeMeasuresSimilarity",
            "secondaryOutcomeMeasuresSimilarity", "overallSimilarity"
        ]

        # Save result to Excel
        result.to_excel(f"{nctNumber}.xlsx", index=False)

        # Convert the DataFrame to a list of dictionaries
        trials_list = result.to_dict(orient="records")

        # Insert response data into the database
        insert_db(
            nctNumber, studyTitle, primaryOutcomeMeasures, secondaryOutcomeMeasures,
            inclusionCriteria, exclusionCriteria, json.dumps(trials_list), conn
        )

        # Return success response
        return JSONResponse(content={"trials": trials_list}, status_code=200)

    except ValueError as e:
        return JSONResponse(content={"message": str(e)}, status_code=400)

    except HTTPException as e:
        raise e  # Let FastAPI handle HTTP exceptions

    except Exception as e:
        return JSONResponse(
            content={"message": "An unexpected error occurred. Please try again later.", "error": str(e)},
            status_code=500
        )

    finally:
        # Close database connection if it was established
        try:
            conn.close()
        except Exception:
            pass
