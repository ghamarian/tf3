$("#slider").roundSlider({
    sliderType: "min-range",
    editableTooltip: false,
    radius: 105,
    width: 16,
    value: 15,
    handleSize: 0,
    handleShape: "square",
    circleShape: "pie",
    startAngle: 315,
    tooltipFormat: "changeTooltip"
});

function changeTooltip(e) {
    var val = e.value, speed;

    return val + "%"
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
