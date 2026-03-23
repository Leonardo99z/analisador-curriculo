from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os
import json
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class AnaliseRequest(BaseModel):
    curriculo: str
    vaga: str

@app.post("/analisar")
def analisar(request: AnaliseRequest):
    prompt = f"""
    Você é um especialista em recrutamento. Analise o currículo abaixo para a vaga descrita.

    CURRÍCULO:
    {request.curriculo}

    VAGA:
    {request.vaga}

    Responda APENAS com um JSON válido, sem texto adicional, neste formato exato:
    {{
      "score": 85,
      "pontos_fortes": ["ponto 1", "ponto 2", "ponto 3"],
      "lacunas": ["lacuna 1", "lacuna 2"],
      "sugestoes": ["sugestão 1", "sugestão 2", "sugestão 3"]
    }}
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024
    )

    texto = response.choices[0].message.content
    resultado = json.loads(texto)

    return {"resultado": resultado}