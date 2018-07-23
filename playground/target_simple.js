$(document).ready(function () {
    table_tag = $('#target');

    table = table_tag.DataTable({
        "columnDefs": [
            {
                "render": function (data, type, row) {
                    return data == -1 ? 'Not relevant' : data;
                },
                "targets": 2
            }
        ],
        'ordering': false,
        'select': 'single'
    });

    $('#submit').prop('disabled', true);

    $('#target tbody').on('click', 'tr', function () {
        if ($(this).hasClass('selected')) {
            $(this).removeClass('selected');
            $('#submit').prop('disabled', true);
        }
        else {
            table.$('tr.selected').removeClass('selected');
            $(this).addClass('selected');
            $('#submit').prop('disabled', false);
        }
    });


    $('form').submit(function () {
        let selected_row = table.row('.selected').data();
        var input = $("<input>")
            .attr("type", "hidden")
            .attr("name", "selected_row").val(JSON.stringify(selected_row));
        $('form').append($(input));
    });
    // $('#select').click(function (event) {
    //     event.preventDefault();
    //     // var cat_column = table.columns(1).data()[0]
    //     let cat_column = table.$('select option:selected').text().split();
    //
    //
    //     $.ajax('/cat_col', {
    //         data: {'cat_column': JSON.stringify(cat_column)},
    //         dataType: 'json',
    //         type: "POST",
    //         // contentType: "application/json; charset=utf-8",
    //         cache: false,
    //         // data: {"a": "b"},
    //         success: function (data) {
    //             $('#next').prop('disabled', false);
    //         }
    //     });
    // });

})
;
