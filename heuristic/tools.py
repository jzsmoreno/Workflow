import glob
import os


def recombine_and_split(pattern):
    # Usar glob para encontrar todos los archivos que coincidan con el patrón
    part_files = glob.glob(pattern)
    part_files.sort()  # Asegurarse de que estén en el orden correcto

    combined_split_lines = []

    # Recorrer cada archivo de partes y agregar sus líneas a combined_split_lines
    for part_file in part_files:
        with open(part_file, "r") as part:
            lines = part.readlines()

            # Filtrar y procesar solo líneas que contengan solo caracteres numéricos
            for line in lines:
                line = line.strip()  # Eliminar espacios en blanco y saltos de línea

                split_line = line.split("\t")  # Dividir la línea por tabuladores

                # Convertir cada valor a un número flotante si es posible
                try:
                    split_line = [float(x) for x in split_line]  # Intentar convertir a float
                except ValueError:
                    # Si ocurre un error en la conversión, se puede optar por dejar el valor original
                    # o manejarlos de alguna manera especial (como poner un valor por defecto).
                    print(f"Advertencia: Línea no convertible a flotantes: {line}")
                    split_line = [
                        None if x == "" else x for x in split_line
                    ]  # O manejarlo como None

                combined_split_lines.append(split_line)

    return combined_split_lines


def split_file(input_file, chunk_size):
    with open(input_file, "r") as f:
        part_num = 0
        buffer = ""
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            buffer += chunk
            lines = buffer.splitlines()
            # If buffer contains enough lines, write them to a part file
            if len(lines) > 1:  # Ensure there is at least one full line
                with open(f"{input_file}_part_{part_num}", "w") as part_file:
                    part_file.write("\n".join(lines))
                buffer = lines[-1]  # Keep the last line as the start of the next chunk
                part_num += 1


def recombine_files(part_files, output_file):
    with open(output_file, "w") as output:
        for part_file in part_files:
            with open(part_file, "r") as part:
                output.write(part.read())


if __name__ == "__main__":
    input_file = "heuristic/evalu.txt"
    split_file(input_file, 50 * 1024 * 1024)  # Dividir en partes de 100MB

    # Define the pattern that matches your part files
    pattern = "heuristic/evalu.txt_part_*"  # Ajustar patrón según el nombre de las partes

    # Obtener el contenido combinado como lista de líneas
    combined_lines_list = recombine_and_split(pattern)

    # Depuración: Verificar cuántas líneas fueron combinadas
    print(f"Total lines combined: {len(combined_lines_list)}")

    breakpoint()
