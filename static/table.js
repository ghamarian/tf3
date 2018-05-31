// $(document).ready(function () {
//     var table = $('#amir').DataTable({
//         'select': 'api'
//     });
//
//     $('#amir').on('click', 'tbody td', function(){
//        // If this column is selected
//        if(table.column(this, { selected: true }).length){
//           table.column(this).deselect();
//
//        // Otherwise, if this column is not selected
//        } else {
//           table.column(this).select();
//        }
//     });
//     // table.on('select', function (e, dt, type, indexes) {
//     //     var data = table.columns(col).data().pluck('id');
//     //     console.log(data)
//     //     // }
//     // });
// })
// ;

var CATEGORIES = CATEGORIES || (function () {
    let _args = {}; // private

    return {
        init: function (Args) {
            _args = Args;
        },

        get: function (cat) {
            return _args[cat];
        }
    };
}());


$(document).ready(function () {

    var category = {
        'categorical': '<select> <option value = "categorical" selected> Categorical </option> <option value = "hash"> Hash </option><option value = "none"> Not used </option></select>',
        'hash': '<select> <option value = "categorical"> Categorical </option> <option value = "hash" selected> Hash </option><option value = "none"> Not used </option></select>',
        'int-range': '<select> <option value = "hash"> Hash </option> <option value = "range" selected> Range </option> <option value = "categorical" > Categorical </option>  <option value = "numerical"> Numerical </option><option value = "none"> Not used </option></select>',
        'int-hash': '<select> <option value = "hash" selected> Hash </option> <option value = "range"> Range </option> <option value = "categorical" > Categorical </option>  <option value = "numerical"> Numerical </option><option value = "none"> Not used </option></select>',
        'int-category': '<select> <option value = "hash"> Hash </option> <option value = "range"> Range </option> <option value = "categorical" selected> Categorical </option>  <option value = "numerical"> Numerical </option><option value = "none"> Not used </option></select>',
        'bool': '<select> <option value = "categorical" selected> Categorical </option><option value = "none"> Not used </option></select>',
        'numerical': '<select> <option value = "numerical" selected> Numerical </option><option value = "none"> Not used </option></select>'
    };

    table_tag = $('#amir');

    table = table_tag.DataTable({
        "columnDefs": [
            {
                "render": function (data, type, row) {
                    return category[data];
                },
                "targets": 1
            },
            {
                "render": function (data, type, row) {
                    return data === -1 ? 'Not relevant' : data;
                },
                "targets": 2
            }
        ],
        'ordering': false,
        // "pageLength": 2
    });


    // console.log($('#category-data').data());


    table_tag.on('page.dt', function () {
        let info = table.page.info();
        $('#pageInfo').html('Showing page: ' + info.page + ' of ' + info.pages);
        let data = table.$('select option:selected').text();
    });

    // $('#send').click(function () {
    //     // var data = table.$('select').serialize();
    //     var data = table.$('select option:selected').text();
    //     alert('Data source: ' + table.columns([0, 1]).dataSrc().join(' '));
    //     alert(
    //         "The following data would have been submitted to the server: \n\n" +
    //         data.substr(0, 120) + '...'
    //     );
    //     return false;
    // });

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


    // var table = $('#amir').DataTable({
    //     'select': 'row'
    // });

    // table.on('select', function (e, dt, type, indexes) {
    //     var data = table.columns(col).data().pluck('id');
    //     console.log(data)
    //     // }
    // });
})
;
