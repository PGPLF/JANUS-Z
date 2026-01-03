#!/bin/bash
# Script de compilation LaTeX pour le document JANUS-JWST

echo "========================================="
echo "Compilation du document JANUS-JWST"
echo "========================================="
echo ""

# Vérifier que pdflatex est installé
if ! command -v pdflatex &> /dev/null; then
    echo "❌ ERREUR: pdflatex n'est pas installé"
    echo ""
    echo "Options:"
    echo "1. Installer BasicTeX:"
    echo "   brew install --cask basictex"
    echo "   eval \"\$(/usr/libexec/path_helper)\""
    echo ""
    echo "2. Ou utiliser Overleaf (voir README_COMPILATION.md)"
    echo ""
    exit 1
fi

echo "✓ pdflatex trouvé"
echo ""

# Aller dans le bon répertoire
cd "$(dirname "$0")" || exit 1

echo "[1/6] Première compilation..."
pdflatex -interaction=nonstopmode janus_jwst_first_results.tex > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Première compilation réussie"
else
    echo "❌ Erreur lors de la première compilation"
    echo "Voir janus_jwst_first_results.log pour détails"
    exit 1
fi

echo "[2/6] Compilation bibliographie..."
if command -v bibtex &> /dev/null; then
    bibtex janus_jwst_first_results > /dev/null 2>&1
    echo "✓ Bibliographie compilée"
else
    echo "⚠ bibtex non trouvé - bibliographie peut être incomplète"
fi

echo "[3/6] Deuxième compilation..."
pdflatex -interaction=nonstopmode janus_jwst_first_results.tex > /dev/null 2>&1
echo "✓ Deuxième compilation réussie"

echo "[4/6] Troisième compilation (résolution références)..."
pdflatex -interaction=nonstopmode janus_jwst_first_results.tex > /dev/null 2>&1
echo "✓ Troisième compilation réussie"

echo "[5/6] Nettoyage des fichiers temporaires..."
rm -f *.aux *.log *.bbl *.blg *.out *.toc *.lof *.lot
echo "✓ Fichiers temporaires supprimés"

echo "[6/6] Vérification du PDF..."
if [ -f "janus_jwst_first_results.pdf" ]; then
    filesize=$(stat -f%z janus_jwst_first_results.pdf)
    echo "✓ PDF généré avec succès!"
    echo ""
    echo "========================================="
    echo "✅ COMPILATION RÉUSSIE"
    echo "========================================="
    echo ""
    echo "Fichier: janus_jwst_first_results.pdf"
    echo "Taille: $(echo "scale=2; $filesize/1024" | bc) Ko"
    echo ""
    echo "Ouvrir le PDF:"
    echo "  open janus_jwst_first_results.pdf"
    echo ""
else
    echo "❌ Erreur: PDF non généré"
    exit 1
fi
