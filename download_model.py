import kagglehub

print("Téléchargement du modèle en cours...")
# Syntaxe complète requise par kagglehub : owner/model-slug/framework/variation
path = kagglehub.model_download("max1234567a/cuneiformbase-400m-akkadian-fn-v3-stage3/transformers/default")

print("Modèle téléchargé avec succès !")
print("Chemin d'accès au modèle :", path)
