import glob
import os

import numpy as np
from numpy import loadtxt


def recombine_and_split(pattern):
    try:
        # Obtener la lista de archivos que coinciden con el patrón
        part_files = glob.glob(pattern)

        # Verificar que se han encontrado archivos
        if not part_files:
            print(f"No se encontraron archivos con el patrón {pattern}")
            return None
        input_file = pattern.replace("_part_*", "")
        recombine_files(pattern, input_file)

        # Convertir la lista de listas a un arreglo NumPy
        combined_array = loadtxt(input_file)

        return combined_array.tolist()

    except Exception as e:
        print(f"Hubo un error al recombinar los archivos: {e}")
        return None


def split_file(input_file, chunk_size):
    with open(input_file, "r") as f:
        part_num = 0
        while True:
            # Leer un chunk de tamaño específico
            chunk = f.read(chunk_size)
            if not chunk:
                break  # Si ya no hay más contenido, terminamos
            # Escribir el chunk en un archivo separado
            with open(f"{input_file}_part_{part_num}", "w") as part_file:
                part_file.write(chunk)
            part_num += 1


def recombine_files(pattern, output_file):
    try:
        # Obtener la lista de archivos que coinciden con el patrón
        part_files = glob.glob(pattern)

        # Verificar que se han encontrado archivos
        if not part_files:
            print(f"No se encontraron archivos con el patrón {pattern}")
            return

        # Ordenar los archivos para asegurarse de que se recombinen en el orden correcto
        part_files.sort()

        # Abrir el archivo de salida en modo escritura
        with open(output_file, "w") as output:
            for part_file in part_files:
                with open(part_file, "r") as part:
                    output.write(part.read())

        print(f"Archivo recombinado guardado en {output_file}")

    except Exception as e:
        print(f"Hubo un error al recombinar los archivos: {e}")


if __name__ == "__main__":
    input_file = "heuristic/evalu.txt"
    # Define the pattern that matches your part files
    pattern = "heuristic/evalu.txt_part_*"  # Ajustar patrón según el nombre de las partes

    split_file(input_file, 50 * 1024 * 1024)  # Dividir en partes de 100MB

    # Obtener el contenido combinado como lista de líneas
    combined_lines_list = recombine_and_split(pattern)

    # Depuración: Verificar cuántas líneas fueron combinadas
    print(f"Total lines combined: {len(combined_lines_list)}")

    breakpoint()
