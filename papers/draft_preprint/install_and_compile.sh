#!/bin/bash
# Script pour installer les packages LaTeX et compiler le PDF
# À exécuter avec: ./install_and_compile.sh

echo "=================================================="
echo "Installation packages LaTeX et compilation PDF"
echo "=================================================="
echo ""

# Vérifier sudo
echo "[1/4] Installation des packages LaTeX manquants..."
echo "  (Votre mot de passe sera demandé pour sudo)"
echo ""

# Mettre à jour tlmgr
sudo /Library/TeX/texbin/tlmgr update --self

# Installer packages requis
sudo /Library/TeX/texbin/tlmgr install collection-fontsrecommended
sudo /Library/TeX/texbin/tlmgr install ec
sudo /Library/TeX/texbin/tlmgr install cm-super

echo ""
echo "✓ Packages installés"
echo ""

# Compiler
echo "[2/4] Compilation du document (passe 1/3)..."
cd "$(dirname "$0")"
/Library/TeX/texbin/pdflatex -interaction=nonstopmode janus_jwst_first_results.tex > compile.log 2>&1

echo "[3/4] Compilation du document (passe 2/3)..."
/Library/TeX/texbin/pdflatex -interaction=nonstopmode janus_jwst_first_results.tex >> compile.log 2>&1

echo "[4/4] Compilation du document (passe 3/3)..."
/Library/TeX/texbin/pdflatex -interaction=nonstopmode janus_jwst_first_results.tex >> compile.log 2>&1

# Vérifier résultat
if [ -f "janus_jwst_first_results.pdf" ]; then
    filesize=$(stat -f%z janus_jwst_first_results.pdf)
    filesize_kb=$(echo "scale=2; $filesize/1024" | bc)

    echo ""
    echo "=================================================="
    echo "✅ PDF GÉNÉRÉ AVEC SUCCÈS!"
    echo "=================================================="
    echo ""
    echo "Fichier: janus_jwst_first_results.pdf"
    echo "Taille: ${filesize_kb} Ko"
    echo ""
    echo "Ouverture du PDF..."
    open janus_jwst_first_results.pdf
    echo ""
else
    echo ""
    echo "❌ Erreur lors de la compilation"
    echo "Voir compile.log pour les détails"
    echo ""
    exit 1
fi
