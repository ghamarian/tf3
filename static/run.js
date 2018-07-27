function add_new_row(key, accuracy, loss, checked) {
    $('#tablediv').append('<tr>')
    if (checked == true) {
        $('#tablediv').append('<td> <input type="radio" id="radiob" name="radiob" value="' + key + '" ' + 'checked><label for="radiob">' + key + '</label>');
    } else {
        $('#tablediv').append('<td> <input type="radio" id="radiob" name="radiob" value="' + key + '"><label for="radiob">' + key + '</label>');
    }
    $('#tablediv').append('<td> ' + accuracy);
    $('#tablediv').append('<td> ' + loss);
    $('#tablediv').append('<td> <a data-id=' + key + ' onclick="ConfirmDelete(this, false)" ><span class="glyphicon glyphicon-remove"></span></a> </tr>');
}

function hide_show() {
    var x = document.getElementById("log");
    if (x.style.display === "none") {
        x.style.display = "block";
        $('#detail_gly').removeClass('glyphicon-triangle-bottom').addClass('glyphicon-triangle-top');
    } else {
        x.style.display = "none";
        $('#detail_gly').removeClass('glyphicon-triangle-top').addClass('glyphicon-triangle-bottom');

    }
}

function hide_show_det() {
    var x = document.getElementById("explain_form");
    if (x.style.display === "none") {
        x.style.display = "block";
        $('#detail_gly2').removeClass('glyphicon-triangle-bottom').addClass('glyphicon-triangle-top');
    } else {
        x.style.display = "none";
        $('#detail_gly2').removeClass('glyphicon-triangle-top').addClass('glyphicon-triangle-bottom');

    }
}

$(document).ready(function () {
    var x = document.getElementById("explain_form");
    x.style.display = "none";
    $('#tablediv').on('click change', 'input:radio[name="radiob"]', function () {
        $('#predict_button').prop('disabled', false);
        $('#explain_button').prop('disabled', false);
        $('#defaultCheck2').prop('disabled', false);
    });

    var output = document.getElementById('log');
    setInterval(function () {
         $.ajax({
            url: "/stream",
            type: 'GET',
            dataType: 'json',
            contentType: 'application/json;charset=UTF-8',
            accepts: {
                json: 'application/json',
            },
            data: JSON.stringify(""),
             success: function(data){

                if (data.data != ''){
                    output.append(data.data)
                    $('#log').scrollTop($('#log')[0].scrollHeight);
                }

            }
         })
    }, 1000);

    setInterval(function () {
        $.ajax({
            url: "/refresh",
            type: 'GET',
            dataType: 'json',
            contentType: 'application/json;charset=UTF-8',
            accepts: {
                json: 'application/json',
            },
            data: JSON.stringify("checkpoints"),
            success: function (data) {
                var $radios = $('input[name="radiob"]');
                var $selected = $radios.filter(':checked');
                if ($selected.val()) {
                    $('#predict_button').prop('disabled', false);
                    $('#explain_button').prop('disabled', false);
                    $('#defaultCheck2').prop('disabled', false);
                } else {
                    $('#predict_button').prop('disabled', true);
                    $('#explain_button').prop('disabled', true);
                    $('#defaultCheck2').prop('disabled', true);
                }
                $('#tablediv').empty();

                $.each(data.checkpoints, function (key, value) {
                    var checked = false;
                    if (key == $selected.val()) {
                        checked = true;
                    }
                    add_new_row(key, value['accuracy'], value['loss'], checked);
                });
            }
        })

    }, 10000)


    $("#predict_button").click(function (e) {
        $.ajax({
            url: "/predict",
            type: 'POST',
            data: $("#predict_form").serialize(),
            success: function (data) {
                if (data.prediction == 'None') {
                    alert('Model\'s structure does not match the new parameter configuration')
                } else {
                    $('#predict_val').text(data.prediction)
                }
            }
        })
    });
    $("#explain_button").click(function (e) {
        $.ajax({
            url: "/explain",
            type: 'POST',
            data: $("#predict_form").serialize(),
            success: function (data) {
                if (data.explanation == 'None') {
                    alert('Model\'s structure does not match the new parameter configuration')
                } else {

                    $("<a>").attr("href", 'explain').attr("target", "_blank")[0].click();

                }
            }
        })
    });

    $("#run_button").click(function (e) {
        var form_data = {'action': 'pause'}
        if (document.getElementById("run_button").className == 'play') {
            form_data['action'] = 'run';
        }
        form_data['resume_from'] = '';
        if (document.getElementById("defaultCheck2").checked == true) {
            var $radios = $('input[name="radiob"]');
            var $selected = $radios.filter(':checked');
            form_data['resume_from'] = $selected.val();
        }
        $.ajax({
            url: "/run",
            type: 'POST',
            data: form_data,
            success: function () {
                $('.play').toggleClass('active');
                $('#defaultCheck2').prop('checked', false);
            }
        })
    });

});

function ConfirmDelete(elem, all) {
    var message = "Are you sure you want to delete the selected model?";
    if (all == true) {
        message = "Are you sure you want to delete all saved models?";
    }
    if (confirm(message)) {
        $.ajax({
            url: "/delete",
            type: 'POST',
            dataType: 'json',
            contentType: 'application/json;charset=UTF-8',
            accepts: {
                json: 'application/json',
            },
            data: JSON.stringify({'deleteID': $(elem).attr('data-id')}),
            success: function (data) {
                if (all == true) {
                    $('#tablediv').empty();
                    $('#predict_button').prop('disabled', true);
                    $('#explain_button').prop('disabled', true);
                    $('#defaultCheck2').prop('disabled', true);
                    document.getElementById("defaultCheck2").checked = false;
                } else {
                    var $radios = $('input[name="radiob"]');
                    var $selected = $radios.filter(':checked');
                    if ($selected.val() && $selected.val() != $(elem).attr('data-id')) {
                        $('#predict_button').prop('disabled', false);
                        $('#explain_button').prop('disabled', false);
                        $('#defaultCheck2').prop('disabled', false);
                    } else {
                        $('#predict_button').prop('disabled', true);
                        $('#explain_button').prop('disabled', true);
                        $('#defaultCheck2').prop('disabled', true);
                        document.getElementById("defaultCheck2").checked = false;
                    }
                    $('#tablediv').empty();

                    $.each(data.checkpoints, function (key, value) {
                        var checked = false;
                        if (key == $selected.val()) {
                            checked = true;
                        }
                        add_new_row(key, value['accuracy'], value['loss'], checked);
                    });
                }

            }
        })
    }
}