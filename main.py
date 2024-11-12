import random

import numpy as np
import copy
import sys
import os


# Convenience methods
def SeparateString(separator, input):
    input = input.strip()
    output = []
    subcomponent = ""
    for char in input:
        if separator == char:
            output.append(subcomponent)
            subcomponent = ""
        else:
            subcomponent += char
    removeList = []
    for i in TableRange(output):  # Removes empty elements
        if output[i] == "":
            removeList.append(i)
    for i in removeList:
        output.pop(i)
    output.append(subcomponent)
    return output


def TableRange(table):
    return range(len(table))


def matrixReference(matrixReferenceName):
    matrixIndex = 0
    for matrix in matrixes:
        if matrix.name == matrixReferenceName:
            return matrixIndex
        matrixIndex += 1
    return None


def askQuestion(question):
    print(question)
    return input()


# Matrix manipulation methods and classes
decimals = 2


class Matrix:
    def __init__(self, name, lines, columns, values=None):
        self.name = name
        self.lines = lines
        self.columns = columns

        if values is None or values == "identity" or values == "null_matrix":
            self.values = []
            self.determinant = None
            for m in range(self.lines):
                self.values.append([])
                for n in range(self.columns):
                    if values is None:
                        self.values[m].append(None)
                    elif values == "identity" and m == n:
                        self.values[m].append(1)
                    else:
                        self.values[m].append(0)

        else:
            self.values = values
            self.updateDeterminant()

    def updateDeterminant(self):
        self.determinant = round(np.linalg.det(self.values), 14)
        if self.determinant % 1 == 0:
            self.determinant = int(self.determinant)
        return self.determinant

    def Print(self):
        printedMatrix = copy.deepcopy(self)
        for m in range(printedMatrix.lines):
            for n in range(printedMatrix.columns):
                printedMatrix.values[m][n] = round(printedMatrix.values[m][n], decimals)
                if printedMatrix.values[m][n] % 1 == 0:
                    printedMatrix.values[m][n] = int(printedMatrix.values[m][n])
        print(printedMatrix.name, ":")
        for m in printedMatrix.values:
            print(m)


matrixes = []


def GenerateDefaultName():
    indexAttempt = 0
    while True:
        defaultName = "m" + str(indexAttempt)
        isValid = True
        for m in matrixes:
            if m.name == defaultName:
                isValid = False
        if isValid:
            return defaultName
        indexAttempt += 1


def initializeMatrix(initializedMatrix):
    print("Définis la matrice: " + initializedMatrix.name)
    for m in range(initializedMatrix.lines):
        validLineInput = False
        while not validLineInput:
            inputList = askQuestion("Tape les " + str(initializedMatrix.columns) + " valeurs à la ligne " + str(
                m + 1) + " (séparées par des espaces)").split()
            if len(inputList) > initializedMatrix.columns:
                print(
                    "Un suplus de valeurs a été fourni, pour s'assurer que les données soient justes, réentrez la bonne quantité de valeurs")
            elif len(inputList) < initializedMatrix.columns:
                print("Quanitié insuffisante de données, réentrez la bonne quantité de valeurs")
            else:
                try:
                    for n in range(initializedMatrix.columns):
                        float(inputList[n])
                    validLineInput = True
                    for n in range(initializedMatrix.columns):
                        initializedMatrix.values[m][n] = float(inputList[n])
                except ValueError:
                    print("Valeur de matrice invalide détectée, réentrez les bonnes valeurs")
    if initializedMatrix.lines == initializedMatrix.columns:
        initializedMatrix.updateDeterminant()


invalid_name_characters = [',', '.', '-', '+', '/', '*', '|', ' ']


def nameMatrix(name):
    if name == "":
        print("Le nom de la matrice ne peut pas être vide, réentrez le nom:")
        return nameMatrix(input())
    for matrix in matrixes:
        if matrix.name == name:
            print("Le même nom de matrice ne peut être utilisé deux fois, réentrez un nom différent:")
            return nameMatrix(input())
    try:
        int(name[0])
        print("Un nom de matrice ne peut pas être débuté par un nombre, réentrez un nom différent:")
        return nameMatrix(input())
    except ValueError:
        for char in name:
            for invalid_char in invalid_name_characters:
                if char == invalid_char:
                    print(
                        "Le nom de matrice contient un caractère invalide -> " + invalid_char + " <-, réentrez un nom différent:")
                    return nameMatrix(input())
    return name


# Matrix Operation methods
def matrixAddition(inputMatrixes):
    matrixSizes = []
    additiveMatrix = None
    for matrix in inputMatrixes:
        if additiveMatrix is None:
            additiveMatrix = copy.deepcopy(matrix)
            additiveMatrix.name = "résultat_somme"
            matrixSizes = [matrix.lines, matrix.columns]
        elif matrix.lines != matrixSizes[0] or matrix.columns != matrixSizes[1]:
            print("Matrices incompatibles pour l'addition, commande cancellée")
            return None
        else:
            for m in range(matrixSizes[0]):
                for n in range(matrixSizes[1]):
                    additiveMatrix.values[m][n] = matrix.values[m][n] + additiveMatrix.values[m][n]
    return additiveMatrix


def matrixScalarMultiplication(coefficient, matrix):
    multipliedMatrix = copy.deepcopy(matrix)
    multipliedMatrix.name = "résultat_somme"
    for m in range(multipliedMatrix.lines):
        for n in range(multipliedMatrix.columns):
            multipliedMatrix.values[m][n] *= coefficient
    return multipliedMatrix


def matrixMultiplication(inputMatrixes):
    multiplicativeMatrix = copy.deepcopy(inputMatrixes[0])
    inputMatrixes.pop(0)
    multiplicativeMatrix.name = "résultat_produit"
    for inputMatrix in inputMatrixes:  # go through all parameter matrices
        if multiplicativeMatrix.columns != inputMatrix.lines:  # if incompatible matrix multiplication return null
            print("Dimensions de matrices incompatibles pour la multiplication matricielle, commande cancellée")
            return None
        else:  # otherwise calculate product from cumulative product and next matrix in list
            resultMatrix = Matrix("résultat_produit", multiplicativeMatrix.lines,
                                  inputMatrix.columns)  # define the next cumulative product
            for m in range(resultMatrix.lines):  # iterate through lines of the next cumulative product
                for n in range(resultMatrix.columns):  # then through columns of each line
                    squareValue = 0
                    for i in range(
                            multiplicativeMatrix.columns):  # iterate from 0 to the amount of terms in the addition in a matrix value
                        squareValue += multiplicativeMatrix.values[m][i] * inputMatrix.values[i][
                            n]  # equation to calculate a term of the addition
                    resultMatrix.values[m][n] = squareValue  # apply the sqaure result to the next cumulative product
            multiplicativeMatrix = copy.deepcopy(resultMatrix)  # define the next cumulative product as the current one
    return multiplicativeMatrix


def matrixExponentiation(inputMatrix, exponent):
    if int(exponent) != exponent:
        print(
            "Cette calculatrice ne supporte pas l'exponentiation fractionnaire des matrices (désolé :/), commande cancellée")
        return None
    exponent = int(exponent)
    if inputMatrix.lines == inputMatrix.columns:
        if exponent == 0:
            return Matrix("résultat_exponentiation", inputMatrix.lines, inputMatrix.columns, "identity")
        if exponent < 0:
            inputMatrix = matrixInversion(inputMatrix)
            exponent = -exponent
        if exponent == 1:
            return copy.deepcopy(inputMatrix)
        matrixMultiplicationList = []
        for i in range(exponent):
            matrixMultiplicationList.append(copy.deepcopy(inputMatrix))
        outputMatrix = matrixMultiplication(matrixMultiplicationList)
        outputMatrix.name = "exponent_result"
        return outputMatrix
    else:
        print(
            "La matrice exponentiée n'est pas carrée, ses dimensions ne sont pas compatibles pour l'exponentiation, commande cancellée")
        return None


def lineOperation(inputMatrix, aIndex, aCoeff, bIndex,
                  bCoeff):  # a prefix means affected line, b prefix means affecting line
    nIndex = 0
    for n in inputMatrix.values[aIndex]:
        inputMatrix.values[aIndex][nIndex] = n * aCoeff - inputMatrix.values[bIndex][nIndex] * bCoeff
        nIndex += 1


def augmentMatrix(inputMatrix, augmenterMatrix):
    augmentedResult = copy.deepcopy(inputMatrix)
    if inputMatrix.lines != augmenterMatrix.lines:
        print("Matrices incompatibles pour l'augmentation")
        return None
    for m in range(inputMatrix.lines):
        for n in augmenterMatrix.values[m]:
            augmentedResult.values[m].append(n)
    augmentedResult.name = "matrice_augmentée"
    augmentedResult.columns = inputMatrix.columns + augmenterMatrix.columns
    return augmentedResult


def matrixInversion(inputMatrix):
    determinant = inputMatrix.determinant
    if inputMatrix.lines != inputMatrix.columns:
        print("Matrice non-carrée, donc non-inversible, commande cancellée")
        return None
    if 0 == determinant:
        print("Déterminant de matrice de 0, donc matrice non-inversible, commande cancellée")
        return None
    size = inputMatrix.lines
    augmentedMatrix = augmentMatrix(inputMatrix, Matrix("Identité_augmentée", size, size, "identity"))
    for n in range(size):
        for m in range(size):
            if m != n:
                lineOperation(augmentedMatrix, m, augmentedMatrix.values[n][n], n, augmentedMatrix.values[m][n])
    for i in range(size):
        lineOperation(augmentedMatrix, i, 1 / augmentedMatrix.values[i][i], i, 0)
    invertedResult = Matrix("matrice_inversée", size, size)
    for m in range(size):
        for n in range(size):
            invertedResult.values[m][n] = augmentedMatrix.values[m][n + size]
    return invertedResult


def Operation(input):
    cleanInput = ""
    recursedMatrixNames = []

    i = 0
    while i < len(input):
        if input[i] == "(":
            recursionInput = ""
            for recursiveChar in input[i + 1:]:
                if recursiveChar == ")":
                    break
                else:
                    recursionInput += recursiveChar
            recursionResult = Operation(recursionInput)
            if recursionResult is None:
                return None
            recursionResult.name = GenerateDefaultName()
            recursedMatrixNames.append(recursionResult.name)
            matrixes.append(recursionResult)
            cleanInput += recursionResult.name
            i += len(recursionInput) + 1
        elif input[i] != " ":
            cleanInput += input[i]
        i += 1
    input = cleanInput

    if cleanInput == "":
        print("Argument ou paranthèse ne peut pas être vide, commande cancellée")
        return None

    i = 0
    while i < len(input):  # Reformatting the unhandled symbols
        if input[i] == "-":
            try:
                if input[i + 1] == "-":
                    input = input[:i] + "+" + input[i + 2:]
                    i -= 1
                elif input[i - 1] == "^":
                    pass
                else:
                    input = input[:i] + "+-1*" + input[i + 1:]
                    i += 4
            except IndexError:
                input = input[:i] + "+"
        elif input[i] == "/":
            nextSymbol = None
            symbolList = ["+", "-", "*", "/", "^"]
            j = 1
            scalar = 1
            while True:
                if i + j >= len(input) - 1:
                    nextSymbol = len(input)
                    break
                try:
                    symbolList.index(input[i + j])
                    nextSymbol = i + j
                    break
                except ValueError:
                    pass
                j += 1
            denominator = input[i + 1:nextSymbol]
            if denominator[0] == "|" and denominator[-1] == "|":
                name = denominator[1:-1]
                reference = matrixReference(name)
                if reference is not None:
                    denominator = matrixes[reference].determinant
                else:
                    print("La matrice \"" + name + "\" n'a pas pu être trouvée, commande cancellée")
                    return None
            else:
                try:
                    denominator = float(denominator)
                except ValueError:
                    print("Le symbole de division doit être suivi d'un nombre, commande cancellée")
                    return None
            try:
                scalar = 1 / denominator
            except ZeroDivisionError:
                print("Division par zéro, commande cancellée")
                return None
            input = input[:i] + "*" + str(scalar) + input[nextSymbol:]
            i += 1 + len(str(scalar))
        i += 1
    draftTables = [[], [], []]
    # Operation prority for addition
    draftTables[0] = SeparateString("+", input)

    # Operation priority for multiplication
    for a in TableRange(draftTables[0]):
        draftTables[1].append(SeparateString("*", draftTables[0][a]))

    # Operation priority for exponentiation
    for a in TableRange((draftTables[1])):
        draftTables[2].append([])
        for b in TableRange(draftTables[1][a]):
            draftTables[2][a].append(SeparateString("^", draftTables[1][a][b]))

    operation = copy.deepcopy(draftTables[-1])
    for a in TableRange(operation):
        for b in TableRange(
                operation[a]):  # Takes care of all the exponent-grade operations and convert all references to matrixes
            if len(operation[a][b]) == 1:  # If the operation does not require exponentiation
                value = operation[a][b][0]
                if value == "":
                    print("La matrice \"" + value + "\" n'a pas pu être trouvée, commande cancellée")
                    return None
                if value[0] == "|" and value[-1] == "|":
                    name = value[1:-1]
                    reference = matrixReference(name)
                    if reference is not None:
                        operation[a][b] = matrixes[reference].determinant
                    else:
                        print("La matrice \"" + name + "\" n'a pas pu être trouvée, commande cancellée")
                        return None
                else:
                    try:
                        operation[a][b] = float(value)
                    except ValueError:
                        reference = matrixReference(value)
                        if reference is not None:
                            operation[a][b] = matrixes[reference]
                        else:
                            print("La matrice \"" + value + "\" n'a pas pu être trouvée, commande cancellée")
                            return None
            else:  # If the operation requires exponentiation
                if len(operation[a][b]) > 2:
                    print(
                        "À des fins de clarté, l'exponentiation d'un exposant doit se faire à l'aide de paranthèses, commande cancellée")
                    return None

                operationDomain = None
                base = operation[a][b][0]
                exponent = operation[a][b][1]

                if base[0] == "|" and base[-1] == "|":  # Manage Base value and operation domain
                    operationDomain = "scalar"
                    name = base[1:-1]
                    reference = matrixReference(name)
                    if reference is not None:
                        base = matrixes[reference].determinant
                    else:
                        print("La matrice \"" + name + "\" n'a pas pu être trouvée, commande cancellée")
                        return None
                else:
                    try:
                        base = float(base)
                        operationDomain = "scalar"
                    except ValueError:
                        operationDomain = "matrix"
                        reference = matrixReference(base)
                        if reference is not None:
                            base = matrixes[reference]
                        else:
                            print("La matrice \"" + base + "\" n'a pas pu être trouvée, commande cancellée")
                            return None

                if exponent[0] == "|" and exponent[-1] == "|":  # Manage exponent value
                    name = exponent[1:-1]
                    reference = matrixReference(name)
                    if reference is not None:
                        exponent = matrixes[reference].determinant
                    else:
                        print("La matrice \"" + name + "\" n'a pas pu être trouvée, commande cancellée")
                        return None
                else:
                    try:
                        exponent = float(exponent)
                    except ValueError:
                        print("Exposant doit être un nombre, commande cancellée")
                        return None

                if operationDomain == "scalar":  # Finding the awnser
                    operation[a][b] = base ** exponent
                elif operationDomain == "matrix":
                    result = matrixExponentiation(base, exponent)
                    if result is not None:
                        operation[a][b] = result
                    else:
                        return None

        # Takes care of all product-grade operations
        scalar = 1
        multiplicationMatrices = []
        numberOutput = False
        for b in TableRange(operation[a]):
            value = operation[a][b]
            if type(value) is float or type(value) is int:
                scalar *= value
            else:
                multiplicationMatrices.append(value)
        if len(multiplicationMatrices) == 0:
            operation[a] = scalar
            numberOutput = True
        elif len(multiplicationMatrices) == 1:
            tamponMatrix = matrixScalarMultiplication(scalar, multiplicationMatrices[0])
        else:
            tamponMatrix = matrixMultiplication(multiplicationMatrices)
            if tamponMatrix is not None:
                tamponMatrix = matrixScalarMultiplication(scalar, tamponMatrix)
        if not numberOutput:
            if tamponMatrix is not None:
                operation[a] = tamponMatrix
            else:
                return None
    # Takes care of all addition-grade operations and returns final matrix value
    isMatrixes = True
    isScalars = True
    for element in operation:
        if type(element) is Matrix:
            isScalars = False
        elif type(element) is float or type(element) is int:
            isMatrixes = False
        else:
            isScalars = False
            isMatrixes = False

    total = None
    if isScalars:
        for element in operation:
            if total is None:
                total = element
            else:
                total += element
    elif isMatrixes:
        total = matrixAddition(operation)
    else:
        print("L'addition entre les scalaires et les matrices n'existe pas, commande cancellée")
        return None

    # Removal of all temporary parentheses matrixes
    for recursedMatrixName in recursedMatrixNames:
        matrixes.pop(matrixReference(recursedMatrixName))

    return total

print("Pour accéder au guide détaillé de commandes, tapez command_list çi-dessous\n")

while True:
    try:
        command = input().split()
        if len(command) != 0:
            if command[0] == "create":
                name = nameMatrix(command[1])
                validEntry = True
                try:
                    command[2]
                    command[1]
                except IndexError:
                    print("La commande create requiert un minimum de deux arguments, tu en as fourni " + str(
                        len(command) - 1) + ", commande cancellée")
                    validEntry = False
                if validEntry:
                    if command[2] == "identity" or command[2] == "Identity":  # Identity matrix preset
                        try:
                            matrixes.append(Matrix(name, int(command[3]), int(command[3]), "identity"))
                            print("Matrice", name, "identité de taille", command[3], "ajoutée")
                        except ValueError:
                            print("Indice d'identité invalide, commande cancellée")
                        except IndexError:
                            print("La création d'une matric identité requiert trois arguments, tu en as fourni " + str(
                                len(command) - 1) + ", commande cancellée")
                    elif command[2] == "null_matrix" or command[2] == "Null_matrix":  # Null matrix preset
                        try:
                            matrixes.append(Matrix(name, int(command[3]), int(command[3]), "null_matrix"))
                            print("Matrice", name, "nulle de taille", command[3], "ajoutée")
                        except ValueError:
                            print("Indice de matrice nulle invalide, commande cancellée")
                        except IndexError:
                            print("la création d'une matrice nulle requiert trois arguments, tu en as fourni " + str(
                                len(command) - 1) + ", commande cancellée")
                    else:  # Custom matrix
                        try:
                            xLocation = command[2].index("x")
                            lines = int(command[2][:xLocation])
                            columns = int(command[2][xLocation + 1:])
                            if lines > 0 and columns > 0:
                                matrixes.append(Matrix(name, lines, columns))
                                initializeMatrix(matrixes[-1])
                            else:
                                print(
                                    "Les dimensions d'une matrice ne peuvent pas être négatives ou nulles, commande cancellée")
                        except ValueError:
                            print("Formattage de dimensions de matrices invalide, commande cancellée")
                print()
            elif command[0] == "remove":
                command.pop(0)
                if len(command) != 0:
                    for element in command:
                        reference = matrixReference(element)
                        if reference is not None:
                            removedMatrixName = matrixes.pop(reference).name
                            print(removedMatrixName + " effacée avec succès")
                        else:
                            print(
                                "La matrice \"" + element + "\" n'a pas pu être trouvée, effacement cancellé pour cette matrice")
                else:
                    matrixes = []
                    print("Les matrices ont toutes étées effacées avec succès")

                print()
            elif command[0] == "rename":
                if len(command) == 3:
                    if matrixReference(command[1]) is not None:
                        matrixes[matrixReference(command[1])].name = command[2]
                        print(command[1] + " matrice renommée avec succès à \"" + command[2] + "\"")
                    else:
                        print("La matrice \"" + command[1] + "\" n'a pas pu être trouvée, commande cancellée")
                else:
                    print("La commande rename requiert deux arguments, tu en as fourni " + str(
                        len(command) - 1) + ", commande cancellée")
                print()
            elif command[0] == "redefine":
                command.pop(0)
                if len(command) != 0:
                    for reference in command:
                        if matrixReference(reference) is not None:
                            initializeMatrix(matrixes[matrixReference(reference)])
                        else:
                            print(
                                "La matrice \"" + reference + "\" n'a pas pu être trouvée, initialisation cancellée pour cette matrice")
                else:
                    print(
                        "la commande redefine requiert au moins un argument, tu en as fourni aucun, commande cancellée")
                print()
            elif command[0] == "display":
                command.pop(0)
                if len(command) != 0:
                    for element in command:
                        if element[0] == "|" and element[-1] == "|":
                            reference = matrixReference(element[1:-1])
                            if reference is not None:
                                print("|" + matrixes[reference].name + "| = " + str(
                                    round(matrixes[reference].determinant, decimals)))
                            else:
                                print("La matrice \"" + element[
                                                        1:-1] + "\" n'a pas pu être trouvée, le déterminant de cette matrice ne sera pas affiché")
                        elif matrixReference(element) is not None:
                            reference = matrixReference(element)
                            matrixes[reference].Print()
                        else:
                            print(
                                "La matrice \"" + element + "\" n'a pas pu être trouvée, cette matrice ne sera pas affichée")
                else:
                    if len(matrixes) == 0:
                        print("Le répertoire de matrices est vide")
                    else:
                        for matrix in matrixes:
                            matrix.Print()

                print()
            elif command[0] == "operate":
                command.pop(0)
                matrixList = []
                matrixName = None
                validEntry = True

                i = 0
                for element in command:  # Finding the name
                    try:  # if it is a name:"" input
                        nameIndex = element.index("name:") + 5
                        matrixName = nameMatrix(element[nameIndex:])
                        command.pop(i)
                        break
                    except ValueError:
                        i += 1

                chainOperation = ""
                for element in command:  # uniting the chain operation under on string
                    chainOperation += element

                finalResult = Operation(chainOperation)

                if finalResult is not None:
                    if type(finalResult) is float or type(finalResult) is int:
                        print("Réponse = " + str(finalResult))
                    else:
                        if finalResult.lines == finalResult.columns:
                            finalResult.updateDeterminant()
                        if matrixName is not None:
                            finalResult.name = matrixName
                        else:
                            finalResult.name = GenerateDefaultName()
                        matrixes.append(finalResult)
                        print("Résultat de l'opération ajouté au répertoire de matrice avec le nom:", finalResult.name)
                print()
            elif command[0] == "decimals":
                validInput = True
                if len(command) == 2:
                    try:
                        Input = float(command[1])
                    except ValueError:
                        print("Le nombre de décimales doit être un nombre, commande cancellée")
                        validInput = False
                    if validInput and int(Input) != Input:
                        print("Le nombre de décimales doit être un nombre entier, commande cancellée")
                        validInput = False
                    elif validInput and Input < 0:
                        print("Le nombre de décimales doit être positif, commande cancellée")
                        validInput = False
                    elif validInput and Input > 14:
                        print("Le nombre maximal de décimales affichables est de 14, commande cancellée")
                        validInput = False
                    if validInput:
                        decimals = int(Input)
                        print("Nombre de décimales affiché changé avec succès à " + str(decimals) + " decimals")
                else:
                    print("La commande decimals requiert un argument, tu en as fourni " + str(
                        len(command) - 1) + ", commande cancellée")
                print()
            elif command[0] == "clear":
                os.system('cls')  # Clears console
                print("Pour accéder au guide détaillé de commandes, tapez command_list si-dessous\n")
            elif command[0] == "command_list":
                print(commandGuide)
            else:
                print("Nom de commande invalide")
                print()
    except KeyboardInterrupt:
        sys.exit(1)
    except Exception as e:
        print("Erreur inconnue, commande intérrompue ; si le programme se comporte de manière anormale, redémarrez-le\nCode d'erreur:")
        print(e)
        print()
