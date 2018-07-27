// function add_new_row(dataset_selected, model, acc, loss) {
//     $('#tablediv').append('<tr> ');
//     $('#tablediv').append('<td> <input type="radio" id="exisiting_files-configuration" name="exisiting_files-configuration" value="' + dataset_selected + '">' + dataset_selected);
//     $('#tablediv').append('</td> <td> ' + model + ' </td> <td> ' + acc + '</td> <td> ' + loss + '</td> ');
//     $('#tablediv').append('<td> <a data-id=' + dataset_selected + ' onclick="ConfirmDelete(this, false)" ><span class="glyphicon glyphicon-remove"></span></a>');
//     $('#tablediv').append(' </tr>');
// };

function add_new_row(dataset_selected, model, acc, loss) {
  var row = '<tr class="tabrow" val="'+ dataset_selected+'" id="' + dataset_selected +'"> <td class="clickable" id="'+ dataset_selected+'" > <input class="invisible-radio" type="radio" id="' + dataset_selected +'b" name="exisiting_files-configuration" value="' + dataset_selected +
      '">' + dataset_selected +'</td> <td class="clickable" id="' + dataset_selected +'"> ' + model + ' </td> <td class="clickable" id="' + dataset_selected +'"> ' + acc + '</td> <td class="clickable" id="' + dataset_selected +'"> ' +  loss + '</td> <td> <a data-id=' + dataset_selected +
      ' onclick="ConfirmDelete(this, false)" ><span class="glyphicon glyphicon-remove"></span></a> </tr>';
    $("#table_config tbody").append(row);
};

function add_new_config_row() {
  var row = '<tr id="new_config" class="tabrow" val="new_config"> <td colspan="5" id="new_config" class="clickable"> <input  class="invisible-radio"  type="radio" id="new_configb" name="exisiting_files-configuration" value="' + 'new_config' +
      '" checked>new config</td> </tr>';
    $("#table_config tbody").append(row);
};

//
// $("input:checkbox").on('click', function() {
//  // in the handler, 'this' refers to the box clicked on
//  var $box = $(this);
//  if ($box.is(":checked")) {
//    // the name of the box is retrieved using the .attr() method
//    // as it is assumed and expected to be immutable
//    var group = "input:checkbox[name='" + $box.attr("name") + "']";
//    // the checked state of the group/box on the other hand will change
//    // and the current value is retrieved using .prop() method
//    $(group).prop("checked", false);
//    $box.prop("checked", true);
//  } else {
//    $box.prop("checked", false);$()
//  }
// });
//
// $('input[type="checkbox"]').on('change', function() {
//   $('input[type="checkbox"]').not(this).prop('checked', false);
// });
