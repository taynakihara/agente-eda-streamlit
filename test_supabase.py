from utils.memoria_db import salvar_memoria, carregar_memoria

# Teste simples de gravação
print("🔹 Testando gravação no Supabase...")

salvar_memoria(
    "Teste de conexão", "Se você está lendo isso, o Supabase está funcionando!", "teste"
)

print("✅ Registro enviado com sucesso!")

# Teste simples de leitura
print("\n🔹 Recuperando últimas interações:")
for item in carregar_memoria(limit=3):
    print(f"{item['timestamp']} → {item['pergunta']} | {item['resposta'][:50]}...")
