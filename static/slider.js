$("#slider").roundSlider({
    sliderType: "min-range",
    editableTooltip: false,
    radius: 105,
    width: 16,
    value: 75,
    handleSize: 0,
    handleShape: "square",
    circleShape: "pie",
    startAngle: 315,
    tooltipFormat: "changeTooltip"
});

function changeTooltip(e) {
    var val = e.value, speed;
    if (val < 20) speed = "Slow";
    else if (val < 40) speed = "Normal";
    else if (val < 70) speed = "Speed";
    else speed = "Very Speed";

    return val + " km/h" + "<div>" + speed + "<div>";
}

// $("#slider").roundSlider({
//     sliderType: "min-range",
//     radius: 130,
//     showTooltip: false,
//     width: 16,
//     value: 46,
//     handleSize: 0,
//     handleShape: "square",
//     circleShape: "half-top"
// });
