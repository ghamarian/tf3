slider = $("#slider").roundSlider({
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

$('form').submit(function () {
    var percent = $('#slider').roundSlider("getValue");
    var input = $("<input>")
        .attr("type", "hidden")
        .attr("name", "percent").val(percent);
    $('form').append($(input));
});


// $('#split').click(function (event) {
//         event.preventDefault();
//         var percent = $('#slider').roundSlider("getValue");
//
//         $.ajax('/split', {
//             data: {'percent': percent},
//             dataType: 'json',
//             type: "POST",
//             cache: false,
//             success: function (data) {
//                 $('#next').prop('disabled', false);
//             }
//         });
//     }
// );

