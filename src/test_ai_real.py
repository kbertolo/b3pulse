from openai_integration import analyze_text_with_openai

print("Chamando OpenAI...")
resp = analyze_text_with_openai("Dê 1 frase de teste dizendo olá", max_tokens=40)
print("Resposta:\n", resp)
