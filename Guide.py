# ANSI escape codes for bold, underline, and reset text
bold_underline_code = '\033[1;4m'
bold_code = '\033[1m'
reset_code = '\033[0m'

commandGuide = f"""{bold_underline_code}Créer un nouvelle matrice:{reset_code}
{bold_code}create "nom de matrice" "nombre de lignes"x"nombre de colonnes" 
OU create "nom de matrice" identity "indice d'identité" 
OU create "nom de matrice" null_matrix "indice de matrice nulle"{reset_code}

{bold_underline_code}Enlever certaines matrices:{reset_code}
{bold_code}remove "nom de matrice #1 (optionel)" "nom de matrice #2 (optionel)" ... "nom de matrice #n (optionel)"{reset_code}
Notes supplémentaires concernant la commande remove:
 - si aucun arguments ne sont données à la commande, elle va effacer toutes les matrices dans le répertoire de l'usager

{bold_underline_code}Renommer une matrice:{reset_code}
{bold_code}rename "nom de matrice" "nouveau nom"{reset_code}

{bold_underline_code}Changer les valeurs de certaines matrices:{reset_code}
{bold_code}redefine "nom de matrice #1" "nom de matrice #2 (optionel)" ... "nom de matrice #n (optionel)"{reset_code}

{bold_underline_code}Afficher certaines matrices:{reset_code}
{bold_code}display "nom de matrice #1 (optionel)" "nom de matrice #2 (optionel)" ... "nom de matrice #n (optionel)"{reset_code}
Notes suppélmentaires concernant la commande display:
 - si aucun arguments ne sont donnés à la commande, elle va afficher toutes les matrices dans le répertoire de l'usager
 - le déterminant d'une matrice peut également être affiché avec les barres verticales (ex: display |A|)

{bold_underline_code}Calculer une équation:{reset_code}
{bold_code}operate "équation" name:"nom de matrice (optionel)"{reset_code}
Type d'opérations acceptées pour l'équation insérée dans la commande operate: 
 - {bold_code}addition:{reset_code} X + Y 
 - {bold_code}soustraction:{reset_code} X - Y 
 - {bold_code}multiplication:{reset_code} X * Y 
 - {bold_code}division:{reset_code} X / Y -- le Y agit comme une multiplication fractionnaire (le dénominateur est limité à des valeurs scalaires)
 - {bold_code}inversion:{reset_code} X^-1 -- cette opération est inclus dans l'exponentiation
 - {bold_code}exponentiation:{reset_code} X^Y -- les exposants négatifs agissent comme une inversion suivie d'une exponentiation
 - {bold_code}déterminant:{reset_code} |X| -- peut être utilisé comme scalaire dans l'équation
exemple d'opération matricielle: {bold_code}operate{reset_code} A + B*C + D^-1 + E^2 + |F|*G name:H -- créé une matrice correspondante au résultat avec le nom \"H\"
exemple d'opération scalaire: {bold_code}operate{reset_code} |A| + |B|*2 + |C|^2 -- affiche le scalaire correspondant au résultat
Notes supplémentaires concernant la commande operate:
 - la priorité des opérations est respectée
 - les espaces n'affectent pas le fonctionnement de l'équation
 - les paranthèses peuvent êtres utilisées pour altérer la priorité des équations, comme en mathématiques traditionelles
 - si la commande ne contient pas d'argument pour le nom, elle va générer un nom par défaut

{bold_underline_code}Ajuster la quantité de décimales affichés:{reset_code}
{bold_code}decimals "nombre de décimals"{reset_code}

{bold_underline_code}Vider la console:{reset_code}
{bold_code}clear{reset_code}
"""