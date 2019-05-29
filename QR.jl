# Iteração QR e método de Householder para matrizes simétricas

using LinearAlgebra, Printf


function qr_iteration(A)
    # Iteração QR
    # para encontrar os autovalores e autovetores de uma matriz simétrica
    n, m = size(A)      # dimensões da matriz A
    tol = 1e-9          # tolerância
    M = 10000            # número máximo de iterações
    convergiu = false   # condição de parada
    cont = 0            # contador de iterações
    Q = Matrix{Float64}(I, n, n)  # preparando a matriz Q

    while convergiu == false && cont <= M
        Qi, R = qr(A)     # decomposição QR
        A = R*Qi
        Q = Q*Qi          # matriz que conterá os autovetores
        cont = cont + 1   # contador de iterações
        k = 0
        for i in 1:n      # iteração nas linhas
            for j in 1:n  # iteração nas colunas
                if abs(A[i, j]) >= tol && i != j  # ainda há elementos >= tol
                    k = 1
                end
            end
        end
        # se k = 0, todas os elementos fora da diagonal da matriz são <= tol
        if k == 0
            convergiu = true
        end
    end

    return A, Q, cont
end


function householder(A)
    # Método de Househoulder
    # transforma uma matriz simétrica em uma matriz simétrica tridiagonal semelhante
    # baseado no algoritmo 9.5 de Numerical Analysis, Burden e Faires (2010)
    n, m = size(A)
    A2 = A
    for k in 1:n - 2 # STEP 1

        q = 0               # STEP 2
        for j in k + 1:n
            q = q + (A[j, k])^2  # soma dos quadrados dos elementos na coluna k
        end

        if A[k + 1, k] == 0     # STEP 3
            alpha = -q^(1/2)  # negativo da norma do vetor da coluna k
        else
            alpha = -q^(1/2)*A[k + 1, k]/abs(A[k + 1, k])
        end

        RSQ = alpha^2 - alpha*A[k + 1, k]    # STEP 4

        v = zeros(n, 1)         # STEP 5
        v[k + 1, 1] = A[k + 1, k] - alpha
        for j in k + 2:n
            v[j, 1] = A[j, k]
        end

        u = zeros(n, 1)         # STEP 6
        for j in k:n
            for i in k + 1:n
                u[j, 1] = u[j, 1] + A[j, i]*v[i, 1]/RSQ
            end
        end

        PROD = 0         # STEP 7
        for i in k + 1:n
            PROD = PROD + v[i, 1]*u[i, 1]
        end

        z = zeros(n, 1)         # STEP 8
        for j in k:n
            z[j, 1] = u[j, 1] - PROD/(2*RSQ)*v[j, 1]
        end

        for l in k + 1:n - 1         # STEP 9

            for j in l + 1:n         # STEP 10
                A2[j, l] = A[j, l] - v[l, 1]*z[j, 1] - v[j, 1]*z[l, 1]
                A2[l, j] = A2[j, l]
            end

            A2[l, l] = A[l, l] - 2*v[l, 1]*z[l, 1]      # STEP 11
        end

        A2[n, n] = A[n, n] - 2*v[n, 1]*z[n, 1]         # STEP 12

        for j in k + 2:n         # STEP 13
            A2[k, j] = 0
            A2[j, k] = 0
        end

        A2[k + 1, k] = A[k + 1, k] - v[k + 1, 1]*z[k, 1]  # STEP 14
        A2[k, k + 1] = A2[k + 1, k]

        A = A2
    end

    return A2
end


function main()
    # teste das funções
    n = 4                # dimensão da matriz
    A = rand(n, n)       # criando uma matriz aleatória n x n
    A = A + A'           # criando uma matriz simétrica
    println("--------------------------------------------------------------")
    println("A = ")
    display(A)
    val, vec = eigen(A)  # autovalores e autovetores da matriz original
    println("autovalores de A = ", val)
    println("autovetores de A = ")
    display(vec)

    A1, Q1, cont1 = qr_iteration(A)
    println("--------------------------------------------------------------")
    println("autovalores de A (iteração QR): ", diag(A1))
    println("autovetores de A (iteração QR): ")
    display(Q1)
    println("Número de iterações (iteração QR): ", cont1)

    H = householder(A)
    println("--------------------------------------------------------------")
    println("matriz tridiagonal semelhante H = ")
    display(H)

    A2, Q2, cont2 = qr_iteration(H)
    println("autovalores de H (householder + iteração QR): ", diag(A2))
    println("Número de iterações (householder + iteração QR): ", cont2)
    println("--------------------------------------------------------------")
end


main()
