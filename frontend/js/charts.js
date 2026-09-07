/**
 * MediFusion AI — Chart.js Helper & Dark Medical Theme Preset
 */

const MediFusionCharts = (function () {
    "use strict";

    const PALETTE = {
        teal: "#2dd4bf",
        tealAlpha: "rgba(45, 212, 191, 0.25)",
        cyan: "#06b6d4",
        cyanAlpha: "rgba(6, 182, 212, 0.25)",
        violet: "#8b5cf6",
        violetAlpha: "rgba(139, 92, 246, 0.25)",
        rose: "#f43f5e",
        roseAlpha: "rgba(244, 63, 94, 0.25)",
        amber: "#f59e0b",
        amberAlpha: "rgba(245, 158, 11, 0.25)",
    };

    const COLORS_LIST = [PALETTE.teal, PALETTE.violet, PALETTE.cyan, PALETTE.amber, PALETTE.rose];
    const COLORS_ALPHA_LIST = [PALETTE.tealAlpha, PALETTE.violetAlpha, PALETTE.cyanAlpha, PALETTE.amberAlpha, PALETTE.roseAlpha];

    // Global Chart.js configuration overrides
    if (window.Chart) {
        Chart.defaults.color = "#0f172a";
        Chart.defaults.borderColor = "rgba(148, 163, 184, 0.4)";
        Chart.defaults.font.family = "'Plus Jakarta Sans', 'Inter', sans-serif";
    }

    /**
     * Render Probability Distribution Bar Chart
     */
    function renderProbabilityBar(canvasId, predictions, horizontal = true) {
        const canvas = document.getElementById(canvasId);
        if (!canvas || !window.Chart) return null;

        const existingChart = Chart.getChart(canvas);
        if (existingChart) existingChart.destroy();

        let labels = [];
        let dataValues = [];

        if (Array.isArray(predictions)) {
            labels = predictions.map((p) => p.disease || p.label || p.name || "Unknown");
            dataValues = predictions.map((p) => {
                const val = p.probability !== undefined ? p.probability : (p.score || 0);
                return (val * 100).toFixed(1);
            });
        } else if (predictions && typeof predictions === "object") {
            labels = Object.keys(predictions);
            dataValues = Object.values(predictions).map((v) => {
                const val = typeof v === "number" ? v : (v.probability || 0);
                return (val * 100).toFixed(1);
            });
        } else {
            return null;
        }

        const bgColors = labels.map((_, i) => COLORS_ALPHA_LIST[i % COLORS_ALPHA_LIST.length]);
        const borderColors = labels.map((_, i) => COLORS_LIST[i % COLORS_LIST.length]);

        return new Chart(canvas, {
            type: "bar",
            data: {
                labels: labels,
                datasets: [
                    {
                        data: dataValues,
                        backgroundColor: bgColors,
                        borderColor: borderColors,
                        borderWidth: 1.5,
                        borderRadius: 6,
                        barPercentage: 0.65,
                    },
                ],
            },
            options: {
                indexAxis: horizontal ? "y" : "x",
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: "rgba(11, 15, 36, 0.95)",
                        titleColor: "#f8fafc",
                        bodyColor: "#cbd5e1",
                        borderColor: "rgba(148, 163, 184, 0.2)",
                        borderWidth: 1,
                        padding: 10,
                        cornerRadius: 8,
                        callbacks: {
                            label: (ctx) => ` Confidence: ${ctx.parsed[horizontal ? "x" : "y"]}%`,
                        },
                    },
                },
                scales: {
                    [horizontal ? "x" : "y"]: {
                        beginAtZero: true,
                        max: 100,
                        grid: { color: "rgba(203, 213, 225, 0.6)" },
                        ticks: {
                            color: "#334155",
                            callback: (v) => `${v}%`,
                            font: { size: 11, weight: "600" },
                        },
                    },
                    [horizontal ? "y" : "x"]: {
                        grid: { display: false },
                        ticks: {
                            font: { size: 12, weight: "700" },
                            color: "#0f172a",
                        },
                    },
                },
                animation: {
                    duration: 900,
                    easing: "easeOutQuart",
                },
            },
        });
    }

    /**
     * Render Modality Radar Chart
     */
    function renderRadarComparison(canvasId, labels, dataPoints) {
        const canvas = document.getElementById(canvasId);
        if (!canvas || !window.Chart) return null;

        const existingChart = Chart.getChart(canvas);
        if (existingChart) existingChart.destroy();

        return new Chart(canvas, {
            type: "radar",
            data: {
                labels: labels,
                datasets: [
                    {
                        label: "Modality Confidence",
                        data: dataPoints,
                        backgroundColor: "rgba(45, 212, 191, 0.15)",
                        borderColor: PALETTE.teal,
                        borderWidth: 2,
                        pointBackgroundColor: PALETTE.teal,
                        pointBorderColor: "#fff",
                        pointRadius: 4,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100,
                        grid: { color: "rgba(203, 213, 225, 0.6)" },
                        angleLines: { color: "rgba(203, 213, 225, 0.6)" },
                        pointLabels: {
                            font: { size: 12, weight: "700" },
                            color: "#0f172a",
                        },
                        ticks: { display: false, stepSize: 25 },
                    },
                },
                animation: { duration: 1000, easing: "easeOutQuart" },
            },
        });
    }

    return {
        renderProbabilityBar,
        renderRadarComparison,
    };
})();

window.MediFusionCharts = MediFusionCharts;
