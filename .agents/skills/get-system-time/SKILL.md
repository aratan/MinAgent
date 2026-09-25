---
name: get-system-time
description: Accede a la hora y fecha actual del sistema, resuelve conversiones de huso horario (ISO 8601/Unix epoch) y calcula deltas de tiempo. Usar siempre que el usuario pregunte por la hora, fecha o cálculos temporales de ejecución.
allowed-tools:
  - bash
  - python_interpreter
---

# Directivas de ejecución de la Skill

Cuando el usuario solicite la hora actual, fecha o cálculos basados en el tiempo del sistema:

1. **Obtención del estado temporal**:
   - Ejecuta preferentemente el script local `scripts/get_time.py` si hay interprete de Python disponible.
   - En su defecto, ejecuta el comando `date -u +"%Y-%m-%dT%H:%M:%SZ"` en la terminal.

2. **Resolución de Zona Horaria**:
   - Si no se especifica huso horario en la consulta, asume el huso del entorno local.
   - Proporciona la salida en formato ISO 8601 (`YYYY-MM-DDTHH:MM:SSZ`) junto con una representación legible para humanos.

3. **Restricciones de respuesta**:
   - NUNCA alucines o asumas la hora basándote en la ventana de contexto o datos de entrenamiento.
   - Si la llamada al sistema falla, notifica el error explícitamente usando la inferencia por omisión: "No puedo verificar la hora actual del sistema en este momento."
