const express = require('express');
const multer = require('multer');
const { exec } = require('child_process');
const path = require('path');
const cors = require('cors');

const app = express();
const PORT = 3000;

app.use(cors());
app.use(express.static('public'));

const upload = multer({ dest: 'uploads/' });

app.post('/api/transcribe', upload.single('audio'), (req, res) => {
    if (!req.file) return res.status(400).json({ error: 'Vui lòng tải file lên.' });

    const audioPath = req.file.path;
    const pythonExec = path.join(__dirname, '.venv', 'Scripts', 'python.exe');
    const scriptPath = path.join(__dirname, 'predict.py');

    console.log(`[Node.js] Đang xử lý file: ${audioPath}`);
    const command = `"${pythonExec}" "${scriptPath}" "${audioPath}"`;
    
    exec(command, (error, stdout, stderr) => {
        if (error) {
            console.error(`[Lỗi] ${error.message}`);
            return res.status(500).json({ error: 'Lỗi chạy mô hình.' });
        }
        res.json({ transcript: stdout.trim() });
    });
});

app.listen(PORT, () => {
    console.log(`🚀 Web đang chạy tại: http://localhost:${PORT}`);
});