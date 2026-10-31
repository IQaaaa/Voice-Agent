import asyncio
import io
import wave

import numpy as np
import streamlit as st

from dotenv import load_dotenv, find_dotenv
from agents import Agent
from agents.voice import (
    AudioInput,
    SingleAgentVoiceWorkflow,
    TTSModelSettings,
    VoicePipeline,
    VoicePipelineConfig,
)


# ============================================================
# CONFIG
# ============================================================

load_dotenv(find_dotenv(), override=True)

SAMPLE_RATE = 24_000

st.set_page_config(
    page_title="Travel Advisor AI",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_prompt" not in st.session_state:
    st.session_state.selected_prompt = None


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    """
    <style>

    /* =====================================================
       GLOBAL
       ===================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 80% 5%,
                rgba(93, 63, 211, 0.28),
                transparent 28%
            ),
            radial-gradient(
                circle at 15% 80%,
                rgba(20, 90, 190, 0.18),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                #020617 0%,
                #07112d 48%,
                #090b24 100%
            );

        color: white;
    }

    .main .block-container {
        max-width: 1500px;
        padding-top: 25px;
        padding-bottom: 50px;
    }

    /* Hide Streamlit branding */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }

    /* =====================================================
       SIDEBAR
       ===================================================== */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #050a25 0%,
                #030719 100%
            );

        border-right: 1px solid rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 25px;
    }

    .brand-icon {
        font-size: 48px;
        margin-bottom: 12px;
    }

    .brand-title {
        font-size: 27px;
        font-weight: 800;
        line-height: 1.1;
        letter-spacing: -1px;
    }

    .brand-gradient {
        background:
            linear-gradient(
                90deg,
                #ffffff,
                #c084fc,
                #60a5fa
            );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .brand-subtitle {
        color: #8290b2;
        font-size: 12px;
        margin-top: 10px;
        line-height: 1.5;
    }

    .side-info {
        margin-top: 35px;
        padding: 20px;

        border-radius: 20px;

        background:
            linear-gradient(
                135deg,
                rgba(99,102,241,0.16),
                rgba(59,130,246,0.05)
            );

        border: 1px solid rgba(139,92,246,0.20);
    }

    .side-info-title {
        font-size: 15px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .side-info-text {
        font-size: 12px;
        color: #8795b7;
        line-height: 1.6;
    }

    .side-footer {
        text-align: center;
        color: #475475;
        font-size: 10px;
        margin-top: 80px;
    }

    /* =====================================================
       TOP BAR
       ===================================================== */

    .topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 22px;
    }

    .welcome-text {
        color: #8492b4;
        font-size: 13px;
    }

    .app-title {
        margin-top: 4px;

        font-size: 32px;
        font-weight: 800;

        background:
            linear-gradient(
                90deg,
                #ffffff,
                #c4b5fd,
                #60a5fa
            );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .online-pill {
        display: flex;
        align-items: center;
        gap: 8px;

        padding: 9px 17px;

        border-radius: 30px;

        background: rgba(34,197,94,0.07);
        border: 1px solid rgba(34,197,94,0.25);

        color: #bbf7d0;
        font-size: 12px;
        font-weight: 600;
    }

    .online-dot {
        width: 8px;
        height: 8px;

        border-radius: 50%;

        background: #22c55e;

        box-shadow:
            0 0 12px #22c55e;
    }

    /* =====================================================
       HERO
       ===================================================== */

    .hero {
        position: relative;
        overflow: hidden;

        min-height: 185px;

        padding: 38px 42px;

        border-radius: 28px;

        background:
            radial-gradient(
                circle at 75% 50%,
                rgba(124,58,237,0.30),
                transparent 32%
            ),
            radial-gradient(
                circle at 15% 50%,
                rgba(37,99,235,0.22),
                transparent 35%
            ),
            rgba(9,16,45,0.82);

        border: 1px solid rgba(255,255,255,0.08);

        box-shadow:
            0 25px 70px rgba(0,0,0,0.28);
    }

    .hero-plane {
        position: absolute;

        right: 7%;
        top: 12px;

        font-size: 115px;

        opacity: 0.07;

        transform: rotate(-12deg);
    }

    .hero-label {
        color: #a78bfa;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.4px;
    }

    .hero-title {
        font-size: 38px;
        font-weight: 800;
        letter-spacing: -1.5px;
        margin-top: 8px;
    }

    .hero-highlight {
        color: #a78bfa;
    }

    .hero-description {
        color: #8998bb;
        font-size: 14px;
        margin-top: 9px;
        max-width: 650px;
    }

    /* =====================================================
       MAIN GRID
       ===================================================== */

    .voice-card {
        margin-top: 22px;

        min-height: 550px;

        padding: 30px;

        border-radius: 28px;

        background:
            radial-gradient(
                circle at 50% 35%,
                rgba(79,70,229,0.14),
                transparent 34%
            ),
            rgba(7,13,38,0.75);

        border: 1px solid rgba(255,255,255,0.08);

        text-align: center;

        box-shadow:
            0 25px 70px rgba(0,0,0,0.22);
    }

    .voice-label {
        color: #7f8eb2;
        font-size: 11px;
        letter-spacing: 1.5px;
        font-weight: 700;
    }

    /* =====================================================
       MICROPHONE
       ===================================================== */

    .mic-container {
        width: 210px;
        height: 210px;

        margin: 35px auto 20px;

        border-radius: 50%;

        display: flex;
        justify-content: center;
        align-items: center;

        background:
            radial-gradient(
                circle,
                #2563eb 0%,
                #4f46e5 42%,
                #7c3aed 70%,
                #101638 72%
            );

        border:
            1px solid rgba(167,139,250,0.9);

        box-shadow:
            0 0 35px rgba(79,70,229,0.7),
            0 0 85px rgba(124,58,237,0.28),
            inset 0 0 30px rgba(255,255,255,0.13);

        animation:
            micPulse 3s ease-in-out infinite;
    }

    @keyframes micPulse {

        0% {
            transform: scale(1);

            box-shadow:
                0 0 30px rgba(79,70,229,0.6),
                0 0 70px rgba(124,58,237,0.20);
        }

        50% {
            transform: scale(1.025);

            box-shadow:
                0 0 50px rgba(79,70,229,0.8),
                0 0 105px rgba(124,58,237,0.34);
        }

        100% {
            transform: scale(1);

            box-shadow:
                0 0 30px rgba(79,70,229,0.6),
                0 0 70px rgba(124,58,237,0.20);
        }
    }

    .mic-symbol {
        font-size: 72px;
        filter: drop-shadow(0 8px 14px rgba(0,0,0,0.4));
    }

    .voice-heading {
        font-size: 21px;
        font-weight: 700;
        margin-top: 15px;
    }

    .voice-subtitle {
        color: #7888ae;
        font-size: 13px;
        margin-top: 7px;
    }

    /* =====================================================
       WAVEFORM
       ===================================================== */

    .waveform {
        height: 65px;

        display: flex;
        justify-content: center;
        align-items: center;

        gap: 5px;

        margin: 22px 0 12px;
    }

    .wave-bar {
        width: 4px;

        border-radius: 10px;

        background:
            linear-gradient(
                180deg,
                #a78bfa,
                #38bdf8
            );

        animation:
            wave 1.2s ease-in-out infinite;
    }

    .wave-bar:nth-child(1) {
        height: 16px;
        animation-delay: .1s;
    }

    .wave-bar:nth-child(2) {
        height: 28px;
        animation-delay: .2s;
    }

    .wave-bar:nth-child(3) {
        height: 43px;
        animation-delay: .3s;
    }

    .wave-bar:nth-child(4) {
        height: 58px;
        animation-delay: .4s;
    }

    .wave-bar:nth-child(5) {
        height: 34px;
        animation-delay: .5s;
    }

    .wave-bar:nth-child(6) {
        height: 50px;
        animation-delay: .6s;
    }

    .wave-bar:nth-child(7) {
        height: 62px;
        animation-delay: .7s;
    }

    .wave-bar:nth-child(8) {
        height: 40px;
        animation-delay: .8s;
    }

    .wave-bar:nth-child(9) {
        height: 55px;
        animation-delay: .9s;
    }

    .wave-bar:nth-child(10) {
        height: 28px;
        animation-delay: 1s;
    }

    .wave-bar:nth-child(11) {
        height: 43px;
        animation-delay: .8s;
    }

    .wave-bar:nth-child(12) {
        height: 20px;
        animation-delay: .6s;
    }

    @keyframes wave {

        0%, 100% {
            transform: scaleY(.65);
            opacity: .55;
        }

        50% {
            transform: scaleY(1);
            opacity: 1;
        }
    }

    /* =====================================================
       AUDIO INPUT
       ===================================================== */

    [data-testid="stAudioInput"] {
        margin: 18px auto;
        max-width: 500px;
    }

    /* =====================================================
       SUGGESTIONS
       ===================================================== */

    .suggestion-label {
        text-align: left;

        color: #7787ac;

        font-size: 11px;

        font-weight: 700;

        letter-spacing: 1px;

        margin-top: 25px;

        margin-bottom: 8px;
    }

    /* =====================================================
       CONVERSATION
       ===================================================== */

    .chat-card {
        margin-top: 22px;

        min-height: 550px;

        padding: 22px;

        border-radius: 28px;

        background:
            linear-gradient(
                145deg,
                rgba(17,25,63,0.88),
                rgba(8,13,35,0.88)
            );

        border: 1px solid rgba(255,255,255,0.08);

        box-shadow:
            0 25px 70px rgba(0,0,0,0.25);
    }

    .chat-header {
        display: flex;
        justify-content: space-between;
        align-items: center;

        padding-bottom: 17px;

        border-bottom:
            1px solid rgba(255,255,255,0.07);
    }

    .chat-title {
        font-size: 18px;
        font-weight: 700;
    }

    .chat-subtitle {
        color: #68779b;
        font-size: 11px;
        margin-top: 3px;
    }

    .empty-chat {
        text-align: center;

        padding: 120px 20px 80px;

        color: #657395;
    }

    .empty-chat-icon {
        font-size: 45px;
        margin-bottom: 15px;
    }

    .empty-chat-title {
        color: #a1aec9;
        font-size: 15px;
        font-weight: 600;
    }

    .empty-chat-text {
        font-size: 12px;
        margin-top: 7px;
    }

    /* =====================================================
       MESSAGE BUBBLES
       ===================================================== */

    .user-message {
        max-width: 88%;

        margin:
            22px
            0
            10px
            auto;

        padding: 13px 16px;

        border-radius:
            18px
            18px
            4px
            18px;

        background:
            linear-gradient(
                135deg,
                #4f46e5,
                #7c3aed
            );

        color: white;

        font-size: 13px;

        box-shadow:
            0 10px 30px rgba(79,70,229,0.18);
    }

    .ai-message {
        max-width: 90%;

        margin:
            10px
            auto
            18px
            0;

        padding: 15px 17px;

        border-radius:
            4px
            18px
            18px
            18px;

        background:
            rgba(255,255,255,0.055);

        border:
            1px solid rgba(255,255,255,0.07);

        color: #dce4f4;

        font-size: 13px;

        line-height: 1.6;
    }

    .message-label {
        color: #8997ba;

        font-size: 9px;

        font-weight: 700;

        letter-spacing: 1px;

        margin-bottom: 6px;
    }

    /* =====================================================
       STATUS
       ===================================================== */

    .ready-card {
        margin-top: 18px;

        padding: 17px 20px;

        border-radius: 18px;

        background:
            linear-gradient(
                90deg,
                rgba(79,70,229,0.13),
                rgba(59,130,246,0.06)
            );

        border:
            1px solid rgba(129,140,248,0.15);

        text-align: left;
    }

    .ready-title {
        font-size: 13px;
        font-weight: 700;
    }

    .ready-text {
        color: #7887aa;
        font-size: 11px;
        margin-top: 3px;
    }

    /* =====================================================
       STREAMLIT BUTTONS
       ===================================================== */

    .stButton > button {

        border-radius: 13px !important;

        min-height: 43px;

        background:
            linear-gradient(
                135deg,
                rgba(79,70,229,0.26),
                rgba(124,58,237,0.20)
            ) !important;

        border:
            1px solid rgba(139,92,246,0.25)
            !important;

        color: white !important;

        font-weight: 600 !important;

        transition:
            all .2s ease
            !important;
    }

    .stButton > button:hover {

        transform:
            translateY(-2px);

        border-color:
            rgba(167,139,250,0.70)
            !important;

        box-shadow:
            0 10px 25px rgba(79,70,229,0.22)
            !important;
    }

    /* =====================================================
       AUDIO PLAYER
       ===================================================== */

    audio {
        width: 100%;
        margin-top: 8px;
    }

    /* =====================================================
       MOBILE
       ===================================================== */

    @media (max-width: 900px) {

        .hero-title {
            font-size: 28px;
        }

        .app-title {
            font-size: 26px;
        }

        .mic-container {
            width: 175px;
            height: 175px;
        }

        .mic-symbol {
            font-size: 58px;
        }

    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.html(
        """
        <div class="brand-icon">
            ✈️
        </div>

        <div class="brand-title">
            <span class="brand-gradient">
                Travel Advisor AI
            </span>
        </div>

        <div class="brand-subtitle">
            Your intelligent voice travel companion
        </div>
        """
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🏠   Home", use_container_width=True):
        pass

    if st.button("🕘   Conversation History", use_container_width=True):
        pass

    if st.button("🧭   Travel Tips", use_container_width=True):
        pass

    if st.button("⚙️   Settings", use_container_width=True):
        pass

    st.html(
        """
        <div class="side-info">

            <div class="side-info-title">
                🌎 Explore the world
            </div>

            <div class="side-info-text">
                Ask your AI travel advisor about
                destinations, itineraries, hotels,
                activities, flights and travel tips.
            </div>

        </div>

        <div class="side-footer">
            Powered by OpenAI Voice Agents
        </div>
        """
    )


# ============================================================
# TOP HEADER
# ============================================================

st.html(
    """
    <div class="topbar">

        <div>
            <div class="welcome-text">
                Your personal AI travel companion
            </div>

            <div class="app-title">
                Travel Advisor AI
            </div>
        </div>

        <div class="online-pill">
            <span class="online-dot"></span>
            AI Online
        </div>

    </div>
    """
)


# ============================================================
# HERO
# ============================================================

st.html(
    """
    <div class="hero">

        <div class="hero-plane">
            ✈️
        </div>

        <div class="hero-label">
            ✨ AI-POWERED VOICE ASSISTANT
        </div>

        <div class="hero-title">
            Your Smart
            <span class="hero-highlight">
                Travel Companion
            </span>
        </div>

        <div class="hero-description">
            Ask anything about travel.
            Discover destinations, plan adventures,
            find hotels and get personalized travel advice —
            simply by speaking.
        </div>

    </div>
    """
)


# ============================================================
# MAIN COLUMNS
# ============================================================

left_column, right_column = st.columns(
    [1.25, 0.85],
    gap="large",
)


# ============================================================
# LEFT — VOICE ASSISTANT
# ============================================================

with left_column:

    st.html(
        """
        <div class="voice-card">

            <div class="voice-label">
                VOICE ASSISTANT
            </div>

            <div class="mic-container">
                <div class="mic-symbol">
                    🎙️
                </div>
            </div>

            <div class="voice-heading">
                Speak to your Travel Advisor
            </div>

            <div class="voice-subtitle">
                Press the microphone below and ask your question naturally.
            </div>

            <div class="waveform">

                <div class="wave-bar"></div>
                <div class="wave-bar"></div>
                <div class="wave-bar"></div>
                <div class="wave-bar"></div>
                <div class="wave-bar"></div>
                <div class="wave-bar"></div>
                <div class="wave-bar"></div>
                <div class="wave-bar"></div>
                <div class="wave-bar"></div>
                <div class="wave-bar"></div>
                <div class="wave-bar"></div>
                <div class="wave-bar"></div>

            </div>

        </div>
        """
    )


    # ========================================================
    # BROWSER MICROPHONE
    # ========================================================

    audio_input_file = st.audio_input(
        "🎙️ Record your question",
        sample_rate=SAMPLE_RATE,
    )


    # ========================================================
    # PROCESS VOICE
    # ========================================================

    if audio_input_file is not None:

        with st.spinner("✨ Travel Advisor is thinking..."):

            try:

                # ------------------------------------------------
                # READ RECORDED AUDIO
                # ------------------------------------------------

                audio_bytes = audio_input_file.getvalue()

                audio_stream = io.BytesIO(
                    audio_bytes
                )

                wav = wave.open(
                    audio_stream,
                    "rb",
                )

                frames = wav.readframes(
                    wav.getnframes()
                )

                channels = wav.getnchannels()

                sample_width = wav.getsampwidth()

                wav.close()


                # ------------------------------------------------
                # CONVERT TO NUMPY
                # ------------------------------------------------

                if sample_width == 2:

                    audio_array = np.frombuffer(
                        frames,
                        dtype=np.int16,
                    )

                elif sample_width == 4:

                    audio_array = (
                        np.frombuffer(
                            frames,
                            dtype=np.int32,
                        )
                        / 65536
                    ).astype(np.int16)

                else:

                    raise ValueError(
                        "Unsupported microphone audio format."
                    )


                # ------------------------------------------------
                # MONO
                # ------------------------------------------------

                if channels > 1:

                    audio_array = audio_array.reshape(
                        -1,
                        channels,
                    )

                    audio_array = (
                        audio_array.mean(axis=1)
                        .astype(np.int16)
                    )


                # ------------------------------------------------
                # AUDIO INPUT FOR AGENTS
                # ------------------------------------------------

                audio_input = AudioInput(
                    buffer=audio_array
                )


                # =================================================
                # AGENT
                # =================================================

                agent = Agent(

                    name="Travel Advisor",

                    instructions=(
                        "You are a friendly, knowledgeable and "
                        "enthusiastic travel advisor. "

                        "Understand the user's spoken travel "
                        "question and answer naturally. "

                        "Keep spoken responses short, useful "
                        "and easy to understand. "

                        "Keep responses to 2-3 sentences maximum."
                    ),

                    model="gpt-5.4-mini",
                )


                # =================================================
                # VOICE PIPELINE
                # =================================================

                pipeline = VoicePipeline(

                    workflow=SingleAgentVoiceWorkflow(
                        agent
                    ),

                    config=VoicePipelineConfig(

                        tts_settings=TTSModelSettings(
                            voice="coral"
                        ),

                        trace_include_sensitive_audio_data=False,
                    ),
                )


                # =================================================
                # PROCESS
                # =================================================

                async def process_voice():

                    result = await pipeline.run(
                        audio_input
                    )

                    response_text = ""

                    audio_chunks = []

                    async for event in result.stream():

                        if (
                            event.type
                            == "voice_stream_event_audio"
                        ):

                            audio_chunks.append(
                                event.data
                            )

                    response_text = (
                        result.total_output_text
                    )

                    return (
                        response_text,
                        audio_chunks,
                    )


                response_text, audio_chunks = asyncio.run(
                    process_voice()
                )


                # =================================================
                # CONVERT AUDIO TO WAV
                # =================================================

                output_audio = None

                if audio_chunks:

                    processed_chunks = []

                    for chunk in audio_chunks:

                        if isinstance(
                            chunk,
                            np.ndarray,
                        ):

                            processed_chunks.append(
                                chunk.astype(np.int16)
                            )

                        elif isinstance(
                            chunk,
                            bytes,
                        ):

                            processed_chunks.append(
                                np.frombuffer(
                                    chunk,
                                    dtype=np.int16,
                                )
                            )


                    if processed_chunks:

                        combined_audio = np.concatenate(
                            processed_chunks
                        ).astype(np.int16)


                        wav_output = io.BytesIO()


                        with wave.open(
                            wav_output,
                            "wb",
                        ) as wav_file:

                            wav_file.setnchannels(1)

                            wav_file.setsampwidth(2)

                            wav_file.setframerate(
                                SAMPLE_RATE
                            )

                            wav_file.writeframes(
                                combined_audio.tobytes()
                            )


                        output_audio = (
                            wav_output.getvalue()
                        )


                # =================================================
                # SAVE MESSAGE
                # =================================================

                st.session_state.messages.append(
                    {
                        "user": "🎙️ Voice question",
                        "assistant": response_text,
                        "audio": output_audio,
                    }
                )


                st.rerun()


            except Exception as error:

                st.error(
                    f"Unable to process the recording: {error}"
                )


    # ========================================================
    # SUGGESTIONS
    # ========================================================

    st.html(
        """
        <div class="suggestion-label">
            ✨ TRY ASKING
        </div>
        """
    )


    suggestion1, suggestion2, suggestion3 = st.columns(3)


    with suggestion1:

        st.button(
            "✈️ Europe destinations",
            use_container_width=True,
        )


    with suggestion2:

        st.button(
            "📸 First-time travel tips",
            use_container_width=True,
        )


    with suggestion3:

        st.button(
            "🏨 Best hotels in Dubai",
            use_container_width=True,
        )


# ============================================================
# RIGHT — CONVERSATION
# ============================================================

with right_column:

    st.html(
        """
        <div class="chat-card">

            <div class="chat-header">

                <div>

                    <div class="chat-title">
                        💬 Conversation
                    </div>

                    <div class="chat-subtitle">
                        Your travel advisor
                    </div>

                </div>

            </div>

        """
    )


    # ========================================================
    # EMPTY CHAT
    # ========================================================

    if not st.session_state.messages:

        st.html(
            """
            <div class="empty-chat">

                <div class="empty-chat-icon">
                    🌎
                </div>

                <div class="empty-chat-title">
                    Your conversation starts here
                </div>

                <div class="empty-chat-text">
                    Ask your first travel question
                    using the microphone.
                </div>

            </div>
            """
        )


    # ========================================================
    # MESSAGES
    # ========================================================

    else:

        for message in st.session_state.messages:

            # User message
            st.html(
                f"""
                <div class="user-message">

                    <div class="message-label">
                        YOU
                    </div>

                    {message["user"]}

                </div>
                """
            )


            # AI message
            st.html(
                f"""
                <div class="ai-message">

                    <div class="message-label">
                        ✨ TRAVEL ADVISOR
                    </div>

                    {message["assistant"]}

                </div>
                """
            )


            # Audio response
            if message.get("audio"):

                st.audio(
                    message["audio"],
                    format="audio/wav",
                )


    st.html(
        """
        </div>
        """
    )


# ============================================================
# READY STATUS
# ============================================================

st.html(
    """
    <div class="ready-card">

        <div class="ready-title">
            ✨ Travel Advisor is ready
        </div>

        <div class="ready-text">
            Speak whenever you're ready. Your AI companion
            will listen, understand and respond.
        </div>

    </div>
    """
)


# ============================================================
# CLEAR CHAT
# ============================================================

if st.session_state.messages:

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(
        [5, 1, 1]
    )

    with col3:

        if st.button(
            "🗑️ Clear",
            use_container_width=True,
        ):

            st.session_state.messages = []

            st.rerun()