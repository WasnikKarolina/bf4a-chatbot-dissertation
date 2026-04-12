// Constants that keep the message types shared between the Wix page and the widget.
const CHATBOT_HTML_ID = "#html1";
const CHATBOT_READY_TYPE = "BF4A_CHATBOT_READY";
const HOST_READY_TYPE = "BF4A_HOST_READY";
const HOST_OPEN_TYPE = "BF4A_OPEN_URL";

$w.onReady(function () {
  // HTML component handle used to exchange postMessage events with the chatbot widget.
  const chatbot = $w(CHATBOT_HTML_ID);

  chatbot.onMessage((event) => {
    // Incoming payload that comes from the embedded chatbot widget.
    const data = event.data || {};

    if (data.type === CHATBOT_READY_TYPE) {
      // Ready handshake that tells the widget the Wix host page can receive actions.
      chatbot.postMessage({ type: HOST_READY_TYPE });
      return;
    }

    if (data.type !== HOST_OPEN_TYPE || !data.payload?.url) {
      return;
    }

    // Open action that lets the widget ask the Wix page to open a target URL.
    const targetUrl = toWindowTarget(data.payload);
    window.open(targetUrl, "_blank", "noopener,noreferrer");
  });
});

function toWindowTarget(payload) {
  // URL helper that resolves relative targets against the current page when needed.
  const rawTarget = payload.target_url || payload.url;

  try {
    // Normalised absolute URL that can be opened safely in a new tab.
    return new URL(rawTarget, window.location.href).toString();
  } catch (error) {
    // Fallback that preserves the raw target if URL parsing fails.
    return rawTarget;
  }
}
