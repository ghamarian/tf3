$(document).ready(function () {
    var table = $('#amir').DataTable({
        'select': 'api'
    });

    $('#amir').on('click', 'tbody td', function(){
       // If this column is selected
       if(table.column(this, { selected: true }).length){
          table.column(this).deselect();

       // Otherwise, if this column is not selected
       } else {
          table.column(this).select();
       }
    });
    // table.on('select', function (e, dt, type, indexes) {
    //     var data = table.columns(col).data().pluck('id');
    //     console.log(data)
    //     // }
    // });
})
;

