import anthropic

client = anthropic.Anthropic()

print(client.models.list(limit=20))