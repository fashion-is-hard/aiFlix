// src.js
// AI flix용 유튜브 영상 리스트

export const videos = [
  // 머신러닝
  {
    id: "ml_freecodecamp_everybody",
    category: "머신러닝",
    title: "Machine Learning for Everybody – Full Course",
    channel: "freeCodeCamp.org",
    youtubeUrl: "https://www.youtube.com/watch?v=i_LwzRVP7bg",
    curation:
      "수학/코딩 경험이 많지 않아도 머신러닝의 전체 흐름을 한 번에 훑어볼 수 있는 입문용 풀코스입니다. TensorFlow를 활용한 실습도 포함되어 있어 개념과 코드 모두를 맛볼 수 있습니다."
  },
  {
    id: "ml_scikit_crash",
    category: "머신러닝",
    title: "Scikit-learn Crash Course - Machine Learning Library for Python",
    channel: "freeCodeCamp.org",
    youtubeUrl: "https://www.youtube.com/watch?v=0B5eIE_1vpU",
    curation:
      "파이썬에서 가장 많이 쓰는 머신러닝 라이브러리인 scikit-learn을 단숨에 정리하는 크래시 코스입니다. 실제 현업 스타일의 전처리·모델링 파이프라인 감을 익히기 좋습니다."
  },
  {
    id: "ml_course_for_beginners",
    category: "머신러닝",
    title: "Machine Learning Course for Beginners",
    channel: "freeCodeCamp.org",
    youtubeUrl: "https://www.youtube.com/watch?v=NWONeJKn6kc",
    curation:
      "지도학습/비지도학습부터 회귀·분류·클러스터링까지, 이론과 실습을 균형 있게 다루는 10시간짜리 머신러닝 정석 입문 강의입니다."
  },
  {
    id: "ml_ws_fullcourse",
    category: "머신러닝",
    title: "Machine Learning FULL Course with Practical (10 HOURS)",
    channel: "WsCube Tech",
    youtubeUrl: "https://www.youtube.com/watch?v=LvC68w9JS4Y",
    curation:
      "인도 쪽 데이터사이언스 커뮤니티에서 많이 보는 코스로, 실습 프로젝트를 따라가며 머신러닝 파이프라인을 몸으로 익힐 수 있는 강의입니다."
  },

  // 딥러닝
  {
    id: "dl_mit_6s191_2024",
    category: "딥러닝",
    title: "MIT Introduction to Deep Learning (2024) | 6.S191",
    channel: "Alexander Amini / MIT Deep Learning",
    youtubeUrl: "https://www.youtube.com/watch?v=ErnWZxJovaM",
    curation:
      "MIT에서 진행하는 딥러닝 집중 강의의 2024년 버전. CNN, RNN, Transformer, 생성모델까지 딥러닝의 정석 커리큘럼을 빠르게 훑고 싶을 때 적합합니다."
  },
  {
    id: "dl_3b1b_neural_net_intro",
    category: "딥러닝",
    title: "But what is a neural network? | Deep learning chapter 1",
    channel: "3Blue1Brown",
    youtubeUrl: "https://www.youtube.com/watch?v=aircAruvnKk",
    curation:
      "뉴럴넷을 처음 접할 때 꼭 봐야 할 시각화 영상. 수식보다 ‘직관’을 먼저 잡고 싶을 때 최고의 선택입니다."
  },
  {
    id: "dl_pytorch_fullcourse",
    category: "딥러닝",
    title: "PyTorch for Deep Learning - Full Course / Tutorial",
    channel: "freeCodeCamp.org",
    youtubeUrl: "https://www.youtube.com/watch?v=GIsg-ZUy0MY",
    curation:
      "PyTorch를 제대로 써보고 싶은 사람을 위한 8시간짜리 풀코스. CNN, GAN까지 실전 예제를 직접 구현하면서 딥러닝 실력을 끌어올릴 수 있습니다."
  },
  {
    id: "dl_korean_neural_kor",
    category: "딥러닝",
    title: "뉴럴네트워크라는걸 들어 보셨다면 보셔야 할 영상 - DL1",
    channel: "3Blue1Brown 한국어",
    youtubeUrl: "https://www.youtube.com/watch?v=wrguEHxk_EI",
    curation:
      "3Blue1Brown 뉴럴넷 시리즈의 한국어 버전으로, 뉴럴넷의 핵심 아이디어를 모국어로 편하게 이해할 수 있습니다."
  },

  // 자연어 처리
  {
    id: "nlp_spacy_fullcourse",
    category: "자연어 처리",
    title: "Natural Language Processing with spaCy & Python - Course for Beginners",
    channel: "freeCodeCamp.org",
    youtubeUrl: "https://www.youtube.com/watch?v=dIUTsFT2MeQ",
    curation:
      "spaCy를 기반으로 토큰화, 품사 태깅, 개체명 인식 등 NLP의 기본기를 실습 중심으로 배우는 입문 강좌입니다."
  },
  {
    id: "nlp_text_analysis_spacy",
    category: "자연어 처리",
    title: "Text Analysis with Python: Intro to spaCy",
    channel: "Data Science Dojo",
    youtubeUrl: "https://www.youtube.com/watch?v=8vAq-SSnMsM",
    curation:
      "짧은 시간에 spaCy를 이용해 텍스트 분석 파이프라인을 구축하는 법을 보여주는 튜토리얼입니다."
  },
  {
    id: "nlp_cs224n_intro",
    category: "자연어 처리",
    title: "CS224N - Intro Lecture (NLP with Deep Learning)",
    channel: "Stanford University",
    youtubeUrl: "https://www.youtube.com/watch?v=rmVRLeJRkl4",
    curation:
      "스탠퍼드 CS224N 강의의 오프닝 강의로, 현대 NLP와 딥러닝의 큰 그림을 잡는 데 도움이 됩니다."
  },
  {
    id: "nlp_tf2_intro",
    category: "자연어 처리",
    title: "Natural Language Processing with TensorFlow 2 - Beginner's Course",
    channel: "freeCodeCamp.org",
    youtubeUrl: "https://www.youtube.com/watch?v=8rXD5-xhemo",
    curation:
      "TensorFlow 2 기반으로 텍스트 분류, 시퀀스 모델링 등을 실습해 보는 입문용 NLP 딥러닝 강좌입니다."
  },

  // 생성형 AI
  {
    id: "genai_freecodecamp_dev",
    category: "생성형 AI",
    title: "Generative AI for Developers – Comprehensive Course",
    channel: "freeCodeCamp.org",
    youtubeUrl: "https://www.youtube.com/watch?v=F0GQ0l2NfHA",
    curation:
      "LLM, 벡터DB, RAG, LangChain, LlamaIndex 등 실무에서 쓰이는 생성형 AI 스택을 한 방에 정리하는 초대형 코스입니다."
  },
  {
    id: "genai_freecodecamp_full",
    category: "생성형 AI",
    title:
      "Generative AI Full Course – Gemini Pro, OpenAI, Llama, Langchain, Pinecone & More",
    channel: "freeCodeCamp.org",
    youtubeUrl: "https://www.youtube.com/watch?v=mEsleV16qdo",
    curation:
      "Gemini, OpenAI, Llama 기반의 실전 프로젝트를 통해 생성형 AI 서비스 전체 구조를 이해할 수 있는 심화 강의입니다."
  },
  {
    id: "genai_3b1b_llm",
    category: "생성형 AI",
    title: "Large Language Models explained briefly",
    channel: "3Blue1Brown",
    youtubeUrl: "https://www.youtube.com/watch?v=LPZh9BOjkQs",
    curation:
      "LLM이 실제로 내부에서 어떻게 동작하는지, 확률모델 관점에서 직관적으로 설명해 주는 영상입니다."
  },
  {
    id: "genai_korean_llm_short",
    category: "생성형 AI",
    title: "LLM 설명 (요약버전)",
    channel: "3Blue1Brown 한국어",
    youtubeUrl: "https://www.youtube.com/watch?v=HnvitMTkXro",
    curation:
      "위 3Blue1Brown LLM 영상을 한국어로 요약·설명한 버전으로, ChatGPT류 모델의 개념을 빠르게 훑기 좋습니다."
  }
];
