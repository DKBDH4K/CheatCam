
async function loadParticles() {
    const { tsParticles } = window;
    
    await tsParticles.load("tsparticles", {
        background: { color: { value: "transparent" } },
        fpsLimit: 60,
        interactivity: {
            events: {
                onHover: { enable: true, mode: "grab" },
                resize: true,
            },
            modes: {
                grab: { distance: 140, links: { opacity: 1 } },
            },
        },
        particles: {
            color: { value: ["#00f3ff", "#ff00e6"] }, 
            links: {
                color: "#00f3ff",
                distance: 150,
                enable: true,
                opacity: 0.2,
                width: 1,
            },
            move: {
                enable: true,
                speed: 1.5,
                direction: "none",
                random: false,
                straight: false,
                outModes: { default: "bounce" },
            },
            number: { value: 120, density: { enable: true, area: 800 } },
            opacity: { value: 0.5 },
            shape: { type: "circle" },
            size: { value: { min: 1, max: 3 } },
        },
        detectRetina: true,
    });
}

loadParticles();

document.addEventListener('DOMContentLoaded', () => {
    const alertLog = document.getElementById('alert-log');
    let lastAlert = "";

    if (alertLog) {
        function addAlertToLog(message, isCritical) {
          
            if (message === lastAlert && message === "Monitoring...") return;
            
            const alertDiv = document.createElement('div');
            alertDiv.className = `alert-item ${isCritical ? 'critical' : ''}`;
            
            const time = new Date().toLocaleTimeString();
            alertDiv.innerHTML = `<strong>[${time}]</strong> ${message}`;
            
            alertLog.prepend(alertDiv);
            lastAlert = message;
        }

      
        setInterval(async () => {
            try {
              
                const response = await fetch('/get_alert');
                const data = await response.json();
                
                if (data.critical || data.alert !== lastAlert) {
                    addAlertToLog(data.alert, data.critical);
                }
            } catch (error) {
                console.error("Waiting for Python Server...", error);
            }
        }, 1000); 
    }
});