#!/bin/bash
# Solution rapide: Installer packages manquants et compiler

echo "================================================================"
echo "Installation packages LaTeX et compilation PDF - Solution rapide"
echo "================================================================"
echo ""
echo "Ce script va installer les packages manquants et compiler le PDF."
echo "Votre mot de passe sudo sera demandé."
echo ""
read -p "Appuyez sur Entrée pour continuer..."

# Installation packages
echo ""
echo "[1/3] Installation collection-fontsrecommended (polices)..."
sudo /Library/TeX/texbin/tlmgr install collection-fontsrecommended

echo ""
echo "[2/3] Installation packages EC fonts..."
sudo /Library/TeX/texbin/tlmgr install ec cm-super

echo ""
echo "[3/3] Compilation du document..."
cd "$(dirname "$0")"

# 3 passes
/Library/TeX/texbin/pdflatex -interaction=nonstopmode janus_jwst_first_results.tex > /dev/null 2>&1
/Library/TeX/texbin/pdflatex -interaction=nonstopmode janus_jwst_first_results.tex > /dev/null 2>&1
/Library/TeX/texbin/pdflatex -interaction=nonstopmode janus_jwst_first_results.tex > /dev/null 2>&1

# Nettoyage
rm -f *.aux *.log *.out *.toc missfont.log

# Vérification
if [ -f "janus_jwst_first_results.pdf" ]; then
    filesize=$(stat -f%z janus_jwst_first_results.pdf)
    filesize_kb=$(echo "scale=2; $filesize/1024" | bc)

    echo ""
    echo "================================================================"
    echo "✅ PDF GÉNÉRÉ AVEC SUCCÈS!"
    echo "================================================================"
    echo ""
    echo "Fichier: janus_jwst_first_results.pdf"
    echo "Taille: ${filesize_kb} Ko"
    echo ""
    open janus_jwst_first_results.pdf
else
    echo ""
    echo "❌ Erreur lors de la compilation"
    echo "Vérifiez janus_jwst_first_results.log"
    exit 1
fi
