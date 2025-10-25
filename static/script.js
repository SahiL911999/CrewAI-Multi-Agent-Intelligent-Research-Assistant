document.addEventListener("DOMContentLoaded", () => {
    const queryForm = document.getElementById("query-form");
    const queryInput = document.getElementById("query-input");
    const chatBox = document.getElementById("chat-box");
    const clearButton = document.getElementById("clear-button");
    const submitButton = document.getElementById("submit-button");
    const micButton = document.getElementById("mic-button");

    let conversationHistory = [];
    let currentSessionId = null;
    let currentAudio = null;

    async function processQuery(query, bypassMemory = false) {
        if (!query) return;
        if (!bypassMemory) { addMessage(query, "user"); conversationHistory.push({ role: "user", content: query }); }
        queryInput.value = ""; submitButton.disabled = true;
        const aiMessageContainer = addMessage("Processing...", "ai");
        try {
            const response = await fetch("/process-query", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ query, history: conversationHistory, bypass_memory: bypassMemory }), });
            currentSessionId = response.headers.get("X-Session-ID");
            if (!response.body) return;
            const reader = response.body.getReader(); const decoder = new TextDecoder(); let finalMessage = "";
            while (true) {
                const { done, value } = await reader.read(); if (done) break;
                const chunk = decoder.decode(value, { stream: true }); const lines = chunk.split('\n\n').filter(line => line.trim() !== '');
                for (const line of lines) {
                    try {
                        const data = JSON.parse(line);
                        if (data.type === 'status') { updateMessage(aiMessageContainer, `*${data.content}*`); }
                        else if (data.type === 'error') { updateMessage(aiMessageContainer, `⚠️ ${data.content}`); }
                        else if (data.type === 'interrupt') { await handleInterrupt(data); }
                        else if (data.type === 'memory_result') { handleMemoryResult(data.content, data.original_query, aiMessageContainer); }
                        else if (data.type === 'result') { finalMessage = data.content.message; }
                    } catch (error) { console.error("Error parsing stream:", error); }
                }
            }
            if (finalMessage) { updateMessage(aiMessageContainer, finalMessage); conversationHistory.push({ role: "assistant", content: finalMessage }); }
        } catch (error) { updateMessage(aiMessageContainer, "❌ An error occurred."); }
        finally { submitButton.disabled = false; }
    }

    queryForm.addEventListener("submit", (e) => { e.preventDefault(); processQuery(queryInput.value.trim(), false); });

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
        const recognition = new SpeechRecognition();
        recognition.continuous = false; recognition.lang = 'en-US'; recognition.interimResults = false; recognition.maxAlternatives = 1;
        micButton.addEventListener('click', () => { micButton.classList.contains('listening') ? recognition.stop() : recognition.start(); });
        recognition.onstart = () => { micButton.classList.add('listening'); micButton.textContent = '...'; };
        recognition.onresult = (event) => { queryInput.value = event.results[0][0].transcript; };
        recognition.onspeechend = () => { recognition.stop(); };
        recognition.onend = () => { micButton.classList.remove('listening'); micButton.textContent = '🎤'; };
        recognition.onerror = (event) => { console.error('Speech recognition error:', event.error); };
    } else { micButton.style.display = 'none'; }

    function handleMemoryResult(content, originalQuery, messageContainer) {
        updateMessage(messageContainer, content); conversationHistory.push({ role: "assistant", content: content });
        const buttonContainer = document.createElement('div'); buttonContainer.className = 'follow-up-buttons';
        const options = {"Sufficient": false, "Search for more": true};
        Object.entries(options).forEach(([text, shouldBypass]) => {
            const btn = document.createElement('button'); btn.className = 'follow-up-btn'; btn.textContent = text;
            btn.addEventListener('click', async () => {
                buttonContainer.innerHTML = `<em>You chose: ${text}</em>`;
                if (shouldBypass) { await processQuery(originalQuery, true); }
            });
            buttonContainer.appendChild(btn);
        });
        messageContainer.appendChild(buttonContainer); chatBox.scrollTop = chatBox.scrollHeight;
    }

    async function handleInterrupt(interruptData) {
        const { message, options } = interruptData;
        const overlay = document.createElement('div'); overlay.className = 'interrupt-overlay';
        overlay.innerHTML = `<div class="interrupt-dialog"><h3>Action Required</h3><p>${message}</p><div class="interrupt-buttons">${options.map(opt => `<button class="interrupt-btn" data-value="${opt}">${opt.toUpperCase()}</button>`).join('')}</div></div>`;
        document.body.appendChild(overlay);
        return new Promise((resolve) => {
            overlay.querySelectorAll('.interrupt-btn').forEach(btn => {
                btn.addEventListener('click', async () => {
                    const choice = btn.getAttribute('data-value');
                    await fetch('/interrupt-response', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session_id: currentSessionId, response: choice }) });
                    document.body.removeChild(overlay); resolve(choice);
                });
            });
        });
    }

    clearButton.addEventListener("click", async () => {
        chatBox.innerHTML = '<div class="message ai-message"><div class="message-content">What can I help you with today?</div></div>';
        conversationHistory = [];
        try { await fetch('/clear-memory', { method: 'POST' }); }
        catch (error) { console.error("Failed to clear server memory:", error); }
    });

    function addMessage(content, type) {
        const messageContainer = document.createElement("div"); messageContainer.className = `message ${type}-message`;
        messageContainer.innerHTML = `<div class="message-content"></div>`;
        messageContainer.querySelector('.message-content').textContent = content;
        chatBox.appendChild(messageContainer); chatBox.scrollTop = chatBox.scrollHeight;
        return messageContainer;
    }

    function updateMessage(container, content) {
        let formattedContent = content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>').replace(/\*(.*?)\*/g, '<em>$1</em>').replace(/\n/g, '<br>');
        container.querySelector(".message-content").innerHTML = formattedContent;
        if (container.classList.contains('ai-message') && content && !content.startsWith('*')) {
            addPlayAudioButton(container, content);
        }
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    function addPlayAudioButton(messageContainer, textToSpeak) {
        if (messageContainer.querySelector('.play-audio-btn')) return;
        const playButton = document.createElement('button'); playButton.className = 'play-audio-btn'; playButton.textContent = '▶️ Play Audio';
        playButton.addEventListener('click', async () => {
            if (currentAudio && !currentAudio.paused) {
                currentAudio.pause(); currentAudio = null; playButton.textContent = '▶️ Play Audio';
                return;
            }
            playButton.textContent = '... Loading'; playButton.disabled = true;
            try {
                const response = await fetch('/synthesize-speech', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: textToSpeak.replace(/<br>/g, '\n').replace(/<[^>]*>/g, '') }) });
                if (!response.ok) throw new Error('Failed to fetch audio.');
                const audioBlob = await response.blob(); const audioUrl = URL.createObjectURL(audioBlob);
                currentAudio = new Audio(audioUrl); currentAudio.play();
                playButton.textContent = '⏹️ Stop'; playButton.disabled = false;
                currentAudio.onended = () => { playButton.textContent = '▶️ Play Audio'; currentAudio = null; };
            } catch (error) { console.error("Error playing audio:", error); playButton.textContent = 'Error'; }
        });
        messageContainer.appendChild(playButton);
    }
});