import numpy as np

def get_matrix_input():
    rows = int(input("Enter the number of rows: "))
    cols = int(input("Enter the number of columns: "))
    matrix = []

    print("Enter the elements row by row:")
    for i in range(rows):
        row = []
        for j in range(cols):
            element = float(input(f"Enter element at ({i + 1}, {j + 1}): "))
            row.append(element)
        matrix.append(row)

    return np.array(matrix)

def matrix_addition(matrix1, matrix2):
    if matrix1.shape != matrix2.shape:
        return None
    return matrix1 + matrix2
def matrix_subtraction(matrix1, matrix2):
    if matrix1.shape != matrix2.shape:
        return None
    return matrix1 - matrix2
def matrix_multiplication(matrix1, matrix2):
    if matrix1.shape[1] != matrix2.shape[0]:
        return None
    return np.dot(matrix1, matrix2)
def main():
    print("Matrix Operations Program")
    print("=========================")

    while True:
        print("\nChoose an operation:")
        print("1. Matrix Addition")
        print("2. Matrix Subtraction")
        print("3. Matrix Multiplication")
        print("4. Exit")

        choice = input("Enter your choice (1/2/3/4): ")

        if choice == '4':
            break

        if choice in ('1', '2', '3'):
            matrix1 = get_matrix_input()
            matrix2 = get_matrix_input()

            if choice == '1':
                result = matrix_addition(matrix1, matrix2)
                operation = "Addition"
            elif choice == '2':
                result = matrix_subtraction(matrix1, matrix2)
                operation = "Subtraction"
            else:
                result = matrix_multiplication(matrix1, matrix2)
                operation = "Multiplication"

            if result is None:
                print("Matrix dimensions are incompatible for the selected operation.")
            else:
                print(f"Result of {operation}:")
                print(result)
        else:
            print("Invalid choice. Please choose a valid operation (1/2/3/4).")


if __name__ == "__main__":
    main()
