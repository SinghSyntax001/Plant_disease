/* ---------------- SAFE DOM READY ---------------- */
document.addEventListener("DOMContentLoaded", () => {

  /* ---------------- NAVIGATION TOGGLE ---------------- */
  const navLinks = document.getElementById("navLinks");

  window.showMenu = function () {
    if (navLinks) navLinks.style.right = "0";
  };

  window.hideMenu = function () {
    if (navLinks) navLinks.style.right = "-200px";
  };

  /* ---------------- UPLOAD MODAL ---------------- */
  let selectedCrop = null;

  const uploadModal = document.getElementById("uploadModal");
  const uploadBtn = document.getElementById("uploadBtn");
  const cropInput = document.getElementById("selectedCrop");
  const imageInput = document.getElementById("imageInput");
  const uploadForm = document.querySelector("form");

  let statusText = document.createElement("div");
  statusText.style.marginTop = "10px";
  statusText.style.fontSize = "14px";
  statusText.style.color = "#2e7d32";

  if (imageInput && imageInput.parentNode) {
    imageInput.parentNode.appendChild(statusText);
  }

  window.openUploadModal = function () {
    if (uploadModal) uploadModal.style.display = "flex";
  };

  window.closeUploadModal = function () {
    if (uploadModal) uploadModal.style.display = "none";
    resetSelection();
  };

  window.selectCrop = function (element) {
    document.querySelectorAll(".modal-item").forEach(item =>
      item.classList.remove("selected")
    );

    element.classList.add("selected");
    selectedCrop = element.querySelector("p")?.innerText || "";

    if (cropInput) cropInput.value = selectedCrop;
    if (uploadBtn) uploadBtn.disabled = true;

    if (imageInput) imageInput.click();
  };

  function resetSelection() {
    selectedCrop = null;
    if (uploadBtn) uploadBtn.disabled = true;
    if (cropInput) cropInput.value = "";
    if (imageInput) imageInput.value = "";
    if (statusText) statusText.innerText = "";
  }

  /* ---------------- FILE SELECTION FEEDBACK ---------------- */
  if (imageInput) {
    imageInput.addEventListener("change", () => {
      if (imageInput.files.length > 0) {
        statusText.innerText = "Selected file: " + imageInput.files[0].name;
        uploadBtn.disabled = false;
      } else {
        statusText.innerText = "";
        uploadBtn.disabled = true;
      }
    });
  }

  /* ---------------- UPLOAD FEEDBACK ---------------- */
  if (uploadForm) {
    uploadForm.addEventListener("submit", () => {
      if (uploadBtn) uploadBtn.disabled = true;
      statusText.innerText = "Uploading image… AI is analyzing the leaf 🌱";
    });
  }

  /* ---------------- CONFIDENCE BAR ---------------- */
  document.querySelectorAll(".confidence-fill").forEach(bar => {
    const conf = bar.dataset.confidence;
    if (conf) {
      bar.style.width = "0%";
      bar.style.transition = "width 1s ease-in-out";
      setTimeout(() => {
        bar.style.width = conf + "%";
      }, 100);
    }
  });

});

/* ================= CHAT LOGIC ================= */

window.sendChat = async function () {
  const input = document.getElementById("chatText");
  const chatBox = document.getElementById("chatMessages");
  const lang = document.getElementById("languageSelect")?.value || "en";

  if (!input || !chatBox || !input.value.trim()) return;

  const userMsg = input.value.trim();
  input.value = "";

  appendMessage("user", userMsg);

  const res = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message: userMsg,
      language: lang,
      context: typeof CHAT_CONTEXT !== "undefined" ? CHAT_CONTEXT : {}
    })
  });

  const data = await res.json();
  appendMessage("bot", data.reply || "No response from AI");
};

function appendMessage(sender, text) {
  const chatBox = document.getElementById("chatMessages");
  const msg = document.createElement("div");
  msg.className = sender === "user" ? "chat-bubble user" : "chat-bubble bot";
  msg.innerText = text;
  chatBox.appendChild(msg);
  chatBox.scrollTop = chatBox.scrollHeight;
}

/* ================= VOICE INPUT (STT) ================= */

let mediaRecorder;
let audioChunks = [];

window.startVoice = async function () {
  audioChunks = [];
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

  mediaRecorder = new MediaRecorder(stream);
  mediaRecorder.start();

  mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
};

window.stopVoice = async function () {
  mediaRecorder.stop();

  mediaRecorder.onstop = async () => {
    const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
    const formData = new FormData();
    formData.append("audio", audioBlob);

    const sttRes = await fetch("/stt", {
      method: "POST",
      body: formData
    });

    const sttData = await sttRes.json();

    document.getElementById("chatText").value = sttData.text || "";
    document.getElementById("languageSelect").value = sttData.language || "en";
  };
};