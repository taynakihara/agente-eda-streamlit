from dotenv import load_dotenv
import os
from supabase import create_client
from datetime import datetime

# ✅ Carregar o .env automaticamente
load_dotenv()

# Pegar as variáveis
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError(
        "As variáveis SUPABASE_URL e SUPABASE_KEY não foram carregadas. Verifique o .env."
    )

# Criar cliente
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def salvar_memoria(pergunta: str, resposta: str, tipo_analise: str = None):
    """
    Armazena a interação do agente no banco de dados Supabase.
    """
    data = {
        "timestamp": datetime.utcnow().isoformat(),
        "pergunta": pergunta,
        "resposta": resposta,
        "tipo_analise": tipo_analise or "geral",
    }
    supabase.table("memoria").insert(data).execute()


def carregar_memoria(limit: int = 10):
    """
    Retorna as últimas interações armazenadas.
    """
    result = (
        supabase.table("memoria")
        .select("*")
        .order("id", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data


def limpar_memoria():
    """
    Remove todos os registros da memória (cuidado!).
    """
    supabase.table("memoria").delete().neq("id", 0).execute()
