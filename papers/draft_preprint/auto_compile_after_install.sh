#!/bin/bash
# Script automatique pour compiler le PDF après installation de BasicTeX

echo "=================================================="
echo "Script de compilation automatique JANUS-JWST"
echo "=================================================="
echo ""

echo "Vérification de l'installation de BasicTeX..."
echo ""

# Boucle d'attente pour l'installation
MAX_WAIT=300  # 5 minutes max
ELAPSED=0
INTERVAL=5

while [ $ELAPSED -lt $MAX_WAIT ]; do
    if [ -f "/Library/TeX/texbin/pdflatex" ]; then
        echo "✓ BasicTeX installé détecté!"
        echo ""
        break
    fi

    echo "  En attente de l'installation... ($ELAPSED s)"
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
done

if [ ! -f "/Library/TeX/texbin/pdflatex" ]; then
    echo "❌ BasicTeX n'est pas installé après $MAX_WAIT secondes"
    echo ""
    echo "Veuillez:"
    echo "1. Terminer l'installation depuis la fenêtre d'installation ouverte"
    echo "2. Ou installer manuellement: sudo installer -pkg /tmp/basictex.pkg -target /"
    echo ""
    exit 1
fi

# Ajouter au PATH
echo "Configuration du PATH..."
export PATH="/Library/TeX/texbin:$PATH"
eval "$(/usr/libexec/path_helper)"

echo ""
echo "[1/5] Mise à jour de tlmgr..."
sudo tlmgr update --self

echo ""
echo "[2/5] Installation des packages LaTeX requis..."
sudo tlmgr install natbib caption booktabs multirow

echo ""
echo "[3/5] Navigation vers le répertoire du document..."
cd "$(dirname "$0")" || exit 1
pwd

echo ""
echo "[4/5] Compilation du document LaTeX..."
echo ""

echo "  [4a] Première passe..."
pdflatex -interaction=nonstopmode janus_jwst_first_results.tex > compile_log.txt 2>&1
if [ $? -ne 0 ]; then
    echo "  ❌ Erreur lors de la compilation"
    echo "  Voir compile_log.txt pour les détails"
    tail -n 50 compile_log.txt
    exit 1
fi
echo "  ✓ Première passe réussie"

echo "  [4b] Compilation bibliographie..."
bibtex janus_jwst_first_results >> compile_log.txt 2>&1
echo "  ✓ Bibliographie compilée"

echo "  [4c] Deuxième passe..."
pdflatex -interaction=nonstopmode janus_jwst_first_results.tex >> compile_log.txt 2>&1
echo "  ✓ Deuxième passe réussie"

echo "  [4d] Troisième passe (résolution références)..."
pdflatex -interaction=nonstopmode janus_jwst_first_results.tex >> compile_log.txt 2>&1
echo "  ✓ Troisième passe réussie"

echo ""
echo "[5/5] Nettoyage et vérification..."
rm -f *.aux *.log *.bbl *.blg *.out *.toc

if [ -f "janus_jwst_first_results.pdf" ]; then
    filesize=$(stat -f%z janus_jwst_first_results.pdf 2>/dev/null || stat -c%s janus_jwst_first_results.pdf)
    filesize_kb=$(echo "scale=2; $filesize/1024" | bc)

    echo ""
    echo "=================================================="
    echo "✅ PDF GÉNÉRÉ AVEC SUCCÈS!"
    echo "=================================================="
    echo ""
    echo "Fichier: janus_jwst_first_results.pdf"
    echo "Taille: ${filesize_kb} Ko"
    echo "Localisation: $(pwd)/janus_jwst_first_results.pdf"
    echo ""
    echo "Ouverture du PDF..."
    open janus_jwst_first_results.pdf
    echo ""
    echo "=================================================="
else
    echo "❌ Erreur: PDF non généré"
    echo "Voir compile_log.txt pour diagnostique"
    exit 1
fi
