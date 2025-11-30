MODEL_CLASSIFIER = "prd5/vit-gambling-finetune"
MODEL_DETECTOR = "prd5/v3-rtdetr-r50-gambling-finetune"

TARGET_CLASSES_FOR_OCR = ("banner_promo", "menu_nav")
CONFIDENCE_THRESHOLD_DETECTOR = 0.1

DEVICE = "cuda"

# OCR Heuristic Constants
GAMBLING_KEYWORDS = [
    # --- CORE JUDI UMUM ---
    "judi", "judi online", "judi bola", "judi slot", "situs judi",
    "situs judi online", "situs judi slot", "bandar judi", "bandar online",
    "agen judi", "bo judi", "bo slot", "bo togel",

    # --- SLOT / RTP / MAXWIN ---
    "slot", "slot online", "slot gacor", "slot gacor hari ini", "situs slot",
    "situs slot online", "situs slot gacor", "slot resmi", "slot terpercaya",
    "game slot", "game slot online", "slot88", "slot303", "slot777",
    "slot maxwin", "maxwin", "max win", "gacor", "gacor maxwin", "rtp",
    "rtp slot", "rtp live", "rtp hari ini", "buy spin", "free spin",
    "scatter", "spin jackpot", "auto spin", "mega win", "big win",
    "super win", "jackpot", "jackpot slot", "jp slot",

    # --- TOGEL / BANDAR TOGEL ---
    "togel", "togel online", "bandar togel", "prediksi togel", "angka jitu",
    "angka main", "colok bebas", "colok macau", "colok naga", "2d", "3d",
    "4d", "result togel", "hasil togel",

    # --- CASINO / LIVE CASINO ---
    "casino", "casino online", "live casino", "live kasino", "roulette",
    "baccarat", "blackjack", "sicbo", "sic bo", "dragon tiger",
    "live roulette", "live baccarat",

    # --- POKER / DOMINO / QQ ---
    "poker", "poker online", "texas poker", "texas holdem", "domino",
    "domino qiuqiu", "qiu qiu", "qiuqiu", "qq", "bandarq", "adu q", "capsa",
    "capsa susun", "ceme", "ceme online",

    # --- SPORTBOOK / JUDI BOLA ---
    "sbobet", "sbo", "sportbook", "sportsbook", "taruhan bola",
    "judi bola online", "parlay", "mix parlay", "handicap", "over under",
    "odds", "live betting", "live bet", "live ball", "bet bola",
    "bola jalan", "score 1x2",

    # --- PROVIDER / BRAND SLOT ---
    "pragmatic", "pragmatic play", "pg soft", "habanero", "joker gaming",
    "spadegaming", "microgaming", "cq9", "rtg slots", "playtech",
    "yggdrasil",

    # --- DEPOSIT / WD / BONUS / PROMO ---
    "deposit", "wd", "withdraw", "withdrawal", "daftar", "daftar slot",
    "daftar judi", "login slot", "login judi", "login member", "akun pro",
    "akun vip", "akun gacor", "slot deposit pulsa", "slot via pulsa",
    "slot depo pulsa", "slot tanpa potongan", "slot tanpa rekening",
    "bonus", "bonus new member", "bonus new member 100",
    "bonus new member 200", "bonus harian", "bonus rollingan",
    "bonus cashback", "cashback", "rollingan", "komisi harian",
    "event slot", "event maxwin", "promo slot", "promo judi", "freebet",
    "free bet", "claim freebet", "claim bonus",

    # --- ISTILAH UMUM / BAIT / AJAKAN ---
    "minimal deposit", "min depo", "depo dana", "depo gopay", "depo ovo",
    "slot dana", "slot gopay", "slot ovo", "anti rungkat", "rungkat",
    "hoki", "cuan slot", "cuan judi", "winrate tinggi", "jam gacor",
    "jam hoki", "link slot", "link alternatif", "link judi", "link resmi",
    "link gacor", "daftar sekarang", "daftar langsung", "main sekarang",
    "mainkan sekarang", "gaskeun slot", "gaskeun judi",

    # --- LAIN-LAIN TERKAIT JUDI ---
    "parlay mix", "slot mudah maxwin", "slot gampang menang",
    "slot gampang jp", "toto togel", "tembak ikan", "fish hunter",
    "dingdong", "mesin slot", "mesin jackpot", "bet", "betting",
    "place bet",
]

KEYWORD_WEIGHTS = {
    # --- CORE JUDI UMUM (TINGGI) ---
    "judi": 3, "judi online": 4, "judi bola": 4, "judi slot": 4,
    "situs judi": 4, "situs judi online": 5, "situs judi slot": 5,
    "bandar judi": 4, "bandar online": 4, "agen judi": 3, "bo judi": 3,
    "bo slot": 3, "bo togel": 3,

    # --- SLOT / RTP / MAXWIN ---
    "slot": 3, "slot online": 4, "slot gacor": 4, "slot gacor hari ini": 5,
    "situs slot": 4, "situs slot online": 5, "situs slot gacor": 5,
    "slot resmi": 3, "slot terpercaya": 3, "game slot": 3,
    "game slot online": 3, "slot88": 4, "slot303": 4, "slot777": 4,
    "slot maxwin": 4, "maxwin": 3, "max win": 3, "gacor": 2,
    "gacor maxwin": 4, "rtp": 2, "rtp slot": 3, "rtp live": 3,
    "rtp hari ini": 3, "buy spin": 2, "free spin": 2, "scatter": 2,
    "spin jackpot": 3, "auto spin": 2, "mega win": 3, "big win": 3,
    "super win": 3, "jackpot": 2, "jackpot slot": 3, "jp slot": 3,

    # --- TOGEL / BANDAR TOGEL ---
    "togel": 3, "togel online": 4, "bandar togel": 4, "prediksi togel": 3,
    "angka jitu": 3, "angka main": 2, "colok bebas": 3, "colok macau": 3,
    "colok naga": 3, "2d": 2, "3d": 2, "4d": 2, "result togel": 3,
    "hasil togel": 3,

    # --- CASINO / LIVE CASINO ---
    "casino": 3, "casino online": 4, "live casino": 4, "live kasino": 4,
    "roulette": 3, "baccarat": 3, "blackjack": 3, "sicbo": 3,
    "sic bo": 3, "dragon tiger": 3, "live roulette": 4, "live baccarat": 4,

    # --- POKER / DOMINO / QQ ---
    "poker": 2, "poker online": 3, "texas poker": 3, "texas holdem": 3,
    "domino": 2, "domino qiuqiu": 3, "qiu qiu": 3, "qiuqiu": 3, "qq": 2,
    "bandarq": 3, "adu q": 3, "capsa": 2, "capsa susun": 3, "ceme": 3,
    "ceme online": 3,

    # --- SPORTBOOK / JUDI BOLA ---
    "sbobet": 4, "sbo": 3, "sportbook": 3, "sportsbook": 3,
    "taruhan bola": 4, "judi bola online": 4, "parlay": 3,
    "mix parlay": 4, "handicap": 2, "over under": 2, "odds": 2,
    "live betting": 3, "live bet": 3, "live ball": 3, "bet bola": 3,
    "bola jalan": 3, "score 1x2": 2,

    # --- PROVIDER / BRAND SLOT ---
    "pragmatic": 3, "pragmatic play": 4, "pg soft": 3, "habanero": 3,
    "joker gaming": 3, "spadegaming": 3, "microgaming": 3, "cq9": 3,
    "rtg slots": 3, "playtech": 3, "yggdrasil": 3,

    # --- DEPOSIT / WD / BONUS / PROMO ---
    "deposit": 1, "wd": 1, "withdraw": 1, "withdrawal": 1, "daftar": 1,
    "daftar slot": 3, "daftar judi": 3, "login slot": 2, "login judi": 2,
    "login member": 1, "akun pro": 2, "akun vip": 2, "akun gacor": 3,
    "slot deposit pulsa": 3, "slot via pulsa": 3, "slot depo pulsa": 3,
    "slot tanpa potongan": 3, "slot tanpa rekening": 3, "bonus": 1,
    "bonus new member": 3, "bonus new member 100": 4,
    "bonus new member 200": 4, "bonus harian": 2, "bonus rollingan": 3,
    "bonus cashback": 3, "cashback": 2, "rollingan": 3,
    "komisi harian": 2, "event slot": 3, "event maxwin": 3,
    "promo slot": 3, "promo judi": 3, "freebet": 3, "free bet": 3,
    "claim freebet": 4, "claim bonus": 3,

    # --- ISTILAH UMUM / BAIT / AJAKAN ---
    "minimal deposit": 2, "min depo": 2, "depo dana": 2, "depo gopay": 2,
    "depo ovo": 2, "slot dana": 3, "slot gopay": 3, "slot ovo": 3,
    "anti rungkat": 2, "rungkat": 2, "hoki": 1, "cuan slot": 3,
    "cuan judi": 3, "winrate tinggi": 2, "jam gacor": 3, "jam hoki": 2,
    "link slot": 3, "link alternatif": 2, "link judi": 3, "link resmi": 2,
    "link gacor": 3, "daftar sekarang": 2, "daftar langsung": 2,
    "main sekarang": 2, "mainkan sekarang": 2, "gaskeun slot": 3,
    "gaskeun judi": 3,

    # --- LAIN-LAIN TERKAIT JUDI ---
    "parlay mix": 3, "slot mudah maxwin": 4, "slot gampang menang": 4,
    "slot gampang jp": 4, "toto togel": 3, "tembak ikan": 3,
    "fish hunter": 3, "dingdong": 2, "mesin slot": 3,
    "mesin jackpot": 3, "bet": 2, "betting": 2, "place bet": 2,
}

SIMILARITY_THRESHOLD = 0.60
MAGIC_NUMBER = 100.0
THRESHOLD_FUSION = 0.5
