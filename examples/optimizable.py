total = 0
total = 1  # asignacion muerta (se pisa sin leerse)

bandera = True

for i in range(4):
    if 2 + 2 == 4:          # condicion constante true
        total = total + i
    else:
        total = total - 999  # inalcanzable

    if False:               # condicion constante false
        basura = i * 100    # inalcanzable
    else:
        basura = 0

    # if anidado para que se vea mas estructura
    if bandera:
        if (3 > 10):        # condicion constante false
            total = total + 1000  # inalcanzable
        else:
            total = total + 1

# while con condicion constante false
while 1 > 2:
    total = total + 50      # inalcanzable

print("total:", total)
