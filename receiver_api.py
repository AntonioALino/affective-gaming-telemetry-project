from flask import Flask, request, jsonify
from datetime import datetime
import json
import csv
import os

app = Flask(__name__)

CSV_FILE = 'heartrate_log.csv'

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['pc_timestamp', 'sensor_timestamp', 'bpm'])


@app.route('/data', methods=['POST'])
def receive_data():
    pc_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    try:
        payload = request.json
    except Exception:
        payload = {}

    print(f"\n[{pc_timestamp}] 📦 Payload recebido: {payload}")

    bpm = None
    sensor_timestamp = pc_timestamp

    if payload and 'value' in payload:
        bpm = payload.get('value')
        sensor_timestamp = payload.get('date', pc_timestamp)

    elif payload and 'payload' in payload:
        eventos = payload.get('payload', [])
        for evento in eventos:
            nome_sensor = evento.get('name', '').lower().replace(' ', '')
            if nome_sensor == 'heartrate':
                valores = evento.get('values', {})
                bpm = valores.get('bpm')
                sensor_timestamp = evento.get('time', pc_timestamp)
                break
    if bpm is not None:
        try:
            with open(CSV_FILE, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([pc_timestamp, sensor_timestamp, bpm])
            print(f"✅ Salvo no CSV: BPM {bpm}")
        except PermissionError:
            print("⚠️ Erro: Arquivo CSV sendo usado por outro programa!")
    else:
        print("⚠️ BPM não encontrado no payload. CSV não preenchido.")

    return jsonify({"status": "success"}), 200


if __name__ == '__main__':
    print(f"🎧 Servidor rodando. Escutando /data. Salvando em: {CSV_FILE}")
    app.run(host='0.0.0.0', port=5000)