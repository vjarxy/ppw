import re
import joblib
import requests
import numpy as np
import streamlit as st

from bs4 import BeautifulSoup


# ============================================================
# KONFIGURASI
# ============================================================

st.set_page_config(
    page_title="Klasifikasi Berita",
    layout="wide"
)


# ============================================================
# CSS
# ============================================================

st.html("""
<style>

html, body {
    background: #f7fbff;
}

.stApp {
    background:
        radial-gradient(
            circle at 8% 85%,
            rgba(87, 191, 225, 0.16),
            transparent 17%
        ),
        radial-gradient(
            circle at 92% 15%,
            rgba(130, 214, 233, 0.10),
            transparent 18%
        ),
        #f7fbff;
}


/* ------------------------------------------------------------
   HILANGKAN DEFAULT STREAMLIT
------------------------------------------------------------ */

#MainMenu {
    visibility: hidden;
}

header {
    visibility: hidden;
}

footer {
    visibility: hidden;
}


/* ------------------------------------------------------------
   CONTENT
------------------------------------------------------------ */

.block-container {
    max-width: 1180px;
    padding-top: 25px;
    padding-bottom: 30px;
}


/* ------------------------------------------------------------
   INPUT
------------------------------------------------------------ */

div[data-testid="stTextInput"] label {
    display: none;
}

div[data-testid="stTextInput"] input {

    height: 40px;

    border-radius: 9px;

    border: 1px solid #dce6ec;

    background: #ffffff;

    color: #455a64;

    font-size: 12px;

    padding-left: 13px;

    box-shadow:
        0 2px 8px rgba(60, 100, 120, 0.04);
}

div[data-testid="stTextInput"] input:focus {

    border-color: #73b7df;

    box-shadow:
        0 0 0 2px rgba(115, 183, 223, 0.12);
}


/* ------------------------------------------------------------
   BUTTON
------------------------------------------------------------ */

div[data-testid="stButton"] button {

    height: 40px;

    border: none;

    border-radius: 9px;

    background: #318ed0;

    color: white;

    font-size: 11px;

    font-weight: 600;

    box-shadow:
        0 5px 12px rgba(49, 142, 208, 0.18);
}

div[data-testid="stButton"] button:hover {

    background: #287fbf;

    color: white;

}


/* ------------------------------------------------------------
   EXPANDER
------------------------------------------------------------ */

div[data-testid="stExpander"] {

    border: 1px solid #e3ebef;

    border-radius: 10px;

    background: white;
}


/* ------------------------------------------------------------
   ALERT
------------------------------------------------------------ */

div[data-testid="stAlert"] {

    border-radius: 10px !important;

    border: 1px solid #d7e3e9 !important;

    box-shadow:
        0 5px 15px rgba(60, 100, 120, 0.08) !important;
}


/* WARNING */

div[data-testid="stAlert"]:has(svg[viewBox="0 0 24 24"]) {

    background-color: #fff8df !important;
}


/* Semua teks alert */

div[data-testid="stAlert"] p {

    color: #43515a !important;

    font-size: 12px !important;

    line-height: 1.5 !important;
}


/* Judul / strong pada alert */

div[data-testid="stAlert"] strong {

    color: #34434c !important;
}

/* ============================================================
   EXPANDER - LIHAT ISI BERITA
============================================================ */

div[data-testid="stExpander"] {

    background: #ffffff !important;

    border: 1px solid #dfe8ed !important;

    border-radius: 10px !important;

    box-shadow:
        0 5px 15px rgba(60, 100, 120, 0.06) !important;
}


/* Judul "Lihat isi berita" */

div[data-testid="stExpander"] summary {

    color: #3f5059 !important;

    font-size: 12px !important;

    font-weight: 600 !important;

}


/* Isi expander */

div[data-testid="stExpander"] [data-testid="stMarkdownContainer"] {

    color: #4f5f67 !important;

    font-size: 12px !important;

    line-height: 1.7 !important;

}


/* Paragraf berita */

div[data-testid="stExpander"] p {

    color: #4f5f67 !important;

    font-size: 12px !important;

    line-height: 1.7 !important;

}


/* Semua teks di dalam expander */

div[data-testid="stExpander"] * {

    color: #4f5f67;
}


/* Tetapi judul expander tetap lebih tegas */

div[data-testid="stExpander"] summary span {

    color: #3f5059 !important;

    font-weight: 600 !important;
}

/* ------------------------------------------------------------
   RESPONSIVE
------------------------------------------------------------ */

@media (max-width: 800px) {

    .block-container {
        padding-left: 20px;
        padding-right: 20px;
    }

}

</style>
""")


# ============================================================
# LOAD MODEL
# ============================================================

model_w2v = joblib.load(
    "model/w2v_model.pkl"
)

model_nb = joblib.load(
    "model/nb_model.pkl"
)


# ============================================================
# PREPROCESSING
# ============================================================

def preprocessing(teks):

    teks = str(teks).lower()

    teks = re.sub(
        r"http\S+|www\S+",
        "",
        teks
    )

    teks = re.sub(
        r"\d+",
        "",
        teks
    )

    teks = re.sub(
        r"[^a-z\s]",
        " ",
        teks
    )

    teks = re.sub(
        r"\s+",
        " ",
        teks
    ).strip()

    return teks


# ============================================================
# DOCUMENT VECTOR
# ============================================================

def document_vector(tokens, model):

    vectors = []

    for word in tokens:

        if word in model.wv:

            vectors.append(
                model.wv[word]
            )

    if len(vectors) == 0:

        return np.zeros(
            model.vector_size
        )

    return np.mean(
        vectors,
        axis=0
    )


# ============================================================
# SCRAPING BERITA
# ============================================================

def ambil_berita_dari_url(url):

    headers = {
        "User-Agent":
        "Mozilla/5.0"
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=15
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    isi = soup.select_one(
        "div.detail__body-text"
    )

    if isi is None:

        isi = soup.select_one(
            "div.detail__body"
        )

    if isi is None:

        isi = soup.select_one(
            "div.itp_bodycontent"
        )

    if isi is None:

        isi = soup.select_one(
            "article"
        )

    if isi is None:

        return None

    return isi.get_text(
        separator=" ",
        strip=True
    )


# ============================================================
# KLASIFIKASI
# ============================================================

def klasifikasi_berita(teks):

    teks_bersih = preprocessing(
        teks
    )

    tokens = teks_bersih.split()

    vektor = document_vector(
        tokens,
        model_w2v
    )

    probabilitas = model_nb.predict_proba(
        [vektor]
    )[0]

    indeks = np.argmax(
        probabilitas
    )

    hasil = model_nb.classes_[indeks]

    confidence = probabilitas[indeks]

    return hasil, confidence


# ============================================================
# VALIDASI URL
# ============================================================

def cek_kategori_url(url):

    url = url.lower()

    if "sport.detik.com" in url:

        return "sport"

    if "finance.detik.com" in url:

        return "finance"

    return None


# ============================================================
# NAVBAR
# ============================================================

nav_col1, nav_col2, nav_col3 = st.columns(
    [1, 1, 8]
)

with nav_col1:

    st.html("""
    <div style="
        font-size:11px;
        color:#263238;
        padding-top:8px;
    ">
        <span style="
            display:inline-block;
            width:7px;
            height:7px;
            background:#f1bd55;
            border-radius:50%;
            margin-right:5px;
        "></span>
        Categories
    </div>
    """)

with nav_col2:

    st.html("""
    <div style="
        font-size:11px;
        color:#718096;
        padding-top:8px;
    ">
        <span style="
            display:inline-block;
            width:7px;
            height:7px;
            background:#75b9df;
            border-radius:50%;
            margin-right:5px;
        "></span>
        Client/Users
    </div>
    """)


# ============================================================
# JARAK
# ============================================================

st.write("")


# ============================================================
# HERO
# ============================================================

left, right = st.columns(
    [1.05, 0.95],
    gap="large"
)


# ============================================================
# BAGIAN KIRI
# ============================================================

with left:

    st.html("""
    <div style="
        padding-top:45px;
    ">

        <div style="
            font-size:43px;
            line-height:1.05;
            font-weight:800;
            color:#202b33;
            letter-spacing:-1.8px;
        ">
            Klasifikasi<br>
            <span style="color:#3998d4;">
                Berita Detik
            </span>
        </div>

        <div style="
            margin-top:15px;
            max-width:430px;
            font-size:13px;
            line-height:1.7;
            color:#89969e;
        ">
            Masukkan URL berita Detik untuk mengetahui
            kategori berita secara otomatis menggunakan
            Skip-gram Word2Vec dan Naive Bayes.
        </div>

        <div style="
            margin-top:25px;
            margin-bottom:7px;
            font-size:10px;
            color:#89969e;
        ">
            URL Berita
        </div>

    </div>
    """)


    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    search_col, button_col = st.columns(
        [4, 1.25],
        gap="small"
    )

    with search_col:

        url = st.text_input(
            "URL Berita",
            placeholder="Masukkan URL berita...",
            label_visibility="collapsed"
        )

    with button_col:

        tombol = st.button(
            "Search",
            use_container_width=True
        )


    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    st.html("""
    <div style="
        margin-top:23px;
        font-size:10px;
        color:#89969e;
    ">
        Features
    </div>

    <div style="
        display:grid;
        grid-template-columns:145px 145px;
        gap:14px;
        margin-top:12px;
    ">

        <div style="
            display:flex;
            align-items:center;
            gap:8px;
            font-size:10px;
            color:#718096;
        ">
            <span style="
                width:27px;
                height:27px;
                display:flex;
                align-items:center;
                justify-content:center;
                border-radius:8px;
                background:#edf7fc;
                color:#4b9bc9;
            ">✦</span>
            Skip-gram
        </div>

        <div style="
            display:flex;
            align-items:center;
            gap:8px;
            font-size:10px;
            color:#718096;
        ">
            <span style="
                width:27px;
                height:27px;
                display:flex;
                align-items:center;
                justify-content:center;
                border-radius:8px;
                background:#edf7fc;
                color:#4b9bc9;
            ">◈</span>
            Naive Bayes
        </div>

        <div style="
            display:flex;
            align-items:center;
            gap:8px;
            font-size:10px;
            color:#718096;
        ">
            <span style="
                width:27px;
                height:27px;
                display:flex;
                align-items:center;
                justify-content:center;
                border-radius:8px;
                background:#edf7fc;
                color:#4b9bc9;
            ">⚽</span>
            Sport
        </div>

        <div style="
            display:flex;
            align-items:center;
            gap:8px;
            font-size:10px;
            color:#718096;
        ">
            <span style="
                width:27px;
                height:27px;
                display:flex;
                align-items:center;
                justify-content:center;
                border-radius:8px;
                background:#edf7fc;
                color:#4b9bc9;
            ">▣</span>
            Finance
        </div>

    </div>
    """)


# ============================================================
# BAGIAN KANAN
# ============================================================

with right:

    st.html("""
    <div style="
        height:440px;
        position:relative;
        display:flex;
        justify-content:center;
        align-items:center;
    ">

        <!-- CIRCLE -->
        <div style="
            position:absolute;
            width:21px;
            height:21px;
            border-radius:50%;
            background:#6bc8df;
            top:45px;
            right:170px;
        "></div>

        <div style="
            position:absolute;
            width:14px;
            height:14px;
            border-radius:50%;
            background:#d65aac;
            top:25px;
            right:128px;
        "></div>

        <div style="
            position:absolute;
            width:28px;
            height:28px;
            border-radius:50%;
            background:#8fd7e5;
            bottom:70px;
            right:55px;
        "></div>

        <div style="
            position:absolute;
            width:15px;
            height:15px;
            border-radius:50%;
            background:#f1bd55;
            bottom:125px;
            left:50px;
        "></div>


        <!-- PHONE -->
        <div style="
            width:178px;
            height:330px;
            border:5px solid #bcc9d0;
            border-radius:29px;
            background:white;
            transform:rotate(13deg);
            box-shadow:
                0 18px 35px rgba(49,91,115,.20);
            position:relative;
            z-index:3;
        ">

            <div style="
                position:absolute;
                top:9px;
                left:60px;
                width:48px;
                height:7px;
                border-radius:8px;
                background:#adb9c0;
            "></div>

            <div style="
                position:absolute;
                top:24px;
                left:9px;
                right:9px;
                bottom:9px;
                border-radius:19px;
                background:
                    linear-gradient(
                        180deg,
                        #def2fa,
                        #ffffff
                    );
            ">

                <div style="
                    padding:15px;
                    color:#428eb8;
                    font-size:9px;
                    font-weight:700;
                ">
                    NEWS
                </div>

                <div style="
                    margin:12px;
                    height:100px;
                    border-radius:12px;
                    background:white;
                    box-shadow:
                        0 5px 14px rgba(50,100,130,.08);
                "></div>

                <div style="
                    position:absolute;
                    top:145px;
                    left:62px;
                    width:43px;
                    height:43px;
                    border-radius:50%;
                    background:#3c99cf;
                    color:white;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    font-size:15px;
                ">
                    ▶
                </div>

            </div>

        </div>


        <!-- FLOATING CARD -->
        <div style="
            position:absolute;
            right:5px;
            top:130px;
            width:92px;
            padding:10px;
            border-radius:10px;
            background:white;
            box-shadow:
                0 9px 24px rgba(48,93,115,.13);
            color:#718096;
            font-size:8px;
            z-index:4;
        ">
            📰<br>
            News Classification
        </div>


        <!-- FLOATING CARD -->
        <div style="
            position:absolute;
            left:35px;
            bottom:85px;
            width:85px;
            padding:10px;
            border-radius:10px;
            background:white;
            box-shadow:
                0 9px 24px rgba(48,93,115,.13);
            color:#718096;
            font-size:8px;
            z-index:4;
        ">
            ✓ Accurate Result
        </div>

    </div>
    """)


# ============================================================
# PROSES KLASIFIKASI
# ============================================================

if tombol:

    if url.strip() == "":

        st.warning(
            "Silakan masukkan URL berita terlebih dahulu."
        )

    else:

        kategori_url = cek_kategori_url(
            url
        )

        # ----------------------------------------------------
        # BUKAN SPORT / FINANCE
        # ----------------------------------------------------

        if kategori_url is None:

            st.warning(
                "Kategori berita tidak didukung."
            )

        else:

            try:

                with st.spinner(
                    "Mengambil dan memproses berita..."
                ):

                    berita = ambil_berita_dari_url(
                        url
                    )

                if berita is None:

                    st.error(
                        "Isi berita tidak ditemukan."
                    )

                else:

                    hasil, confidence = klasifikasi_berita(
                        berita
                    )

                    # ------------------------------------------------
                    # HASIL
                    # ------------------------------------------------

                    if hasil.lower() == "sport":

                        kategori = "🏆 SPORT"

                    else:

                        kategori = "💰 FINANCE"


                    st.html(
                        f"""
                        <div style="
                            margin-top:25px;
                            padding:18px 22px;
                            background:white;
                            border:1px solid #e3ebef;
                            border-radius:12px;
                            box-shadow:
                                0 6px 20px
                                rgba(60,100,120,.06);
                        ">

                            <div style="
                                font-size:10px;
                                color:#89959d;
                            ">
                                Hasil Klasifikasi
                            </div>

                            <div style="
                                margin-top:3px;
                                font-size:23px;
                                font-weight:800;
                                color:#398fc6;
                            ">
                                {kategori}
                            </div>

                            <div style="
                                margin-top:3px;
                                font-size:10px;
                                color:#89959d;
                            ">
                            </div>

                        </div>
                        """
                    )


                    # ------------------------------------------------
                    # ISI BERITA
                    # ------------------------------------------------

                    with st.expander(
                        "Lihat isi berita"
                    ):

                        st.write(
                            berita
                        )

            except requests.exceptions.RequestException:

                st.error(
                    "URL tidak dapat diakses. "
                    "Pastikan URL berita Detik benar."
                )

            except Exception as e:

                st.error(
                    f"Terjadi kesalahan: {e}"
                )


# ============================================================
# FOOTER
# ============================================================

st.html("""
<div style="
    text-align:center;
    margin-top:25px;
    color:#a4afb5;
    font-size:9px;
">
    Klasifikasi Berita Detik •
    Skip-gram Word2Vec + Naive Bayes
</div>
""")