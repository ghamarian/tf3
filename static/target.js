
$(document).ready(function () {

    table_tag = $('#target');

    table = table_tag.DataTable({
        "columnDefs": [
            {
                "render": function (data, type, row) {
                    return data === -1 ? 'Not relevant' : data;
                },
                "targets": 2
            }
        ],
        'ordering': false,
        'select': 'single'
    });

    $('#select').click(function (event) {
        event.preventDefault();
        // var cat_column = table.columns(1).data()[0]
        let cat_column = table.$('select option:selected').text().split();


        $.ajax('/cat_col', {
            data: {'cat_column': JSON.stringify(cat_column)},
            dataType: 'json',
            type: "POST",
            // contentType: "application/json; charset=utf-8",
            cache: false,
            // data: {"a": "b"},
            success: function (data) {
                $('#next').prop('disabled', false);
            }
        });
    });

})
;
