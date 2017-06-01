import json,csv
import xlsxwriter
import xlrd

class Store(object):
    def import_profil_to_excel(cls,header_format,filename,data,sheet_name):
        """
        :param header_format: berupa list contoh : [1,2,3]
        :param filename: nama file
        :param data: berupa dictionary dengan aturan value berupa list contoh key:[1,2,3]
        :param sheet_name: nama dari sheet nya
        :return:
        """
        workbook = xlsxwriter.Workbook(filename=filename+".xlsx")
        profil_sheet = workbook.add_worksheet(sheet_name)

        c = 0
        for header in header_format:
            profil_sheet.write(0,c,header)
            c+=1

        row = 1
        col = 0
        for trait,data_col in data.items():
            profil_sheet.write(row,col,trait)
            for count_col in range(1,len(data_col)+1):
                profil_sheet.write(row,count_col,data_col[count_col-1])
            row+=1

        workbook.close()
        #trick biar bisa edit di file yang sama...
        wbRD = xlrd.open_workbook("{}.xlsx".format(filename))
        sheets = wbRD.sheets()

        workbook = xlsxwriter.Workbook(filename=filename + ".xlsx")

        for sheet in sheets:  # write data from old file
            newSheet = workbook.add_worksheet(sheet.name)
            for row in range(sheet.nrows):
                for col in range(sheet.ncols):
                    newSheet.write(row, col, sheet.cell(row, col).value)
        row = 1
        col = 0
        for trait,data_col in data.items():
            newSheet.write(row,col,trait)
            for count_col in range(1,len(data_col)+1):
                newSheet.write(row,count_col,data_col[count_col-1])
            row+=1
        print("Import profile to excel succes...")

